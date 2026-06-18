"""
Derivative-based Noisy Binary Search for Hoprate Adjustment
============================================================

A variation of the Multiplicative Weights Update (MWU) noisy binary search
algorithm that uses the *derivative* (rate of change) metric based on BER
and hoprate, instead of simple BER increase/decrease, to make directional decisions.

Core idea: instead of using a binary "BER went up/down" signal, we calculate:

    metric = ΔBER_percent / Δhoprate_clamped

Where:
  - ΔBER_percent = (current_BER - previous_BER) × 100
    (BER change in percentage points, e.g., 0.1 → 0.02 gives ΔBER_percent = -8)
  - Δhoprate_clamped = current_hoprate - previous_hoprate
    If Δhoprate is exactly 0, it is clamped to -0.01 to avoid division by zero.

And use this continuous metric to decide which half of the search space to
reinforce.  This uses more information from each observation than the original
binary comparison.

Decision rule (configurable threshold, default -0.002):

  metric > threshold   →  answer = "target is to the LEFT"
                          → h ≤ h_curr compatible  →  w × 2(1−p)
                          → h > h_curr incompatible →  w × 2p

  metric ≤ threshold   →  answer = "target is to the RIGHT"
                          → h ≥ h_curr compatible  →  w × 2(1−p)
                          → h < h_curr incompatible →  w × 2p

Special case: if Δhoprate is exactly 0, it is clamped to -0.01 to avoid division
by zero and provide a deterministic default direction.

The rest of the algorithm (weight normalisation, weighted-median query
selection, convergence criterion) is identical to the original implementation.
"""

import numpy as np
from typing import Tuple, Optional


class DerivativeNoisyBinarySearch:
    """
    Derivative-based noisy binary search over a hoprate range [hoprate_min, hoprate_max].

    Uses Δhoprate / ΔBER derivative instead of simple BER comparison to make
    directional decisions.

    Parameters
    ----------
    hoprate_min : float
        Minimum candidate hoprate (Hz).
    hoprate_max : float
        Maximum candidate hoprate (Hz).
    hoprate_step : float
        Discretisation step (Hz).  Default 10 — matches ``_apply_hoprate``.
    p : float
        Assumed noise probability, 0 ≤ p < 0.5.  Smaller p → faster convergence
        but less tolerance to noisy BER readings.
    delta : float
        Confidence / convergence threshold, 0 < δ ≤ 1.  When the maximum weight
        reaches 1 − δ the algorithm reports convergence.
    derivative_threshold : float
        Decision threshold for the derivative metric.  Default -0.002.
        metric > threshold → LEFT move; metric ≤ threshold → RIGHT move.
    seed : int or None
        RNG seed for reproducible query randomisation.
    """

    def __init__(
        self,
        hoprate_min: float = 10.0,
        hoprate_max: float = 1000.0,
        hoprate_step: float = 10.0,
        p: float = 0.1,
        delta: float = 0.01,
        derivative_threshold: float = -0.002,
        seed: Optional[int] = None,
    ):
        if not (0.0 <= p < 0.5):
            raise ValueError(f"p must be in [0, 0.5), got {p}")
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {delta}")

        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.hoprate_step = float(hoprate_step)
        self.p = float(p)
        self.delta = float(delta)
        self.derivative_threshold = float(derivative_threshold)

        # ---- candidate grid ---------------------------------------------------
        n = int(round((self.hoprate_max - self.hoprate_min) / self.hoprate_step)) + 1
        self.candidates = np.linspace(
            self.hoprate_min, self.hoprate_max, n, dtype=np.float64
        )
        self.n_candidates = len(self.candidates)

        # ---- internal state ---------------------------------------------------
        self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates

        self._current_idx: Optional[int] = None
        self._previous_idx: Optional[int] = None
        self._current_hoprate: Optional[float] = None
        self._previous_hoprate: Optional[float] = None
        self._previous_ber: Optional[float] = None
        self._step_count: int = 0

        self._rng = np.random.RandomState(seed)

        # ---- diagnostic state -------------------------------------------------
        self._last_derivative: Optional[float] = None
        self._last_delta_hoprate: Optional[float] = None
        self._last_delta_ber: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> float:
        """
        Reset all internal state and return the initial hoprate to test.

        Returns
        -------
        float
            The first hoprate selected by the (uniform) weighted median.
        """
        self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates
        self._current_idx = None
        self._previous_idx = None
        self._current_hoprate = None
        self._previous_hoprate = None
        self._previous_ber = None
        self._step_count = 0
        self._last_derivative = None
        self._last_delta_hoprate = None
        self._last_delta_ber = None
        return self._select_hoprate()

    def seed_observation(self, hoprate: float, ber: float) -> None:
        """
        Seed the algorithm with an initial (hoprate, BER) pair.

        Use this when the first tested hoprate is chosen externally (not via
        ``reset()``).  After calling, the next ``step()`` will compare its BER
        against this seed and update weights accordingly.

        Parameters
        ----------
        hoprate : float
            The hoprate that was actually tested.
        ber : float
            The mean BER observed at that hoprate.
        """
        idx = int(np.argmin(np.abs(self.candidates - float(hoprate))))
        self._previous_idx = self._current_idx
        self._previous_hoprate = self._current_hoprate
        self._current_idx = idx
        self._current_hoprate = float(self.candidates[idx])
        self._previous_ber = float(ber)

    def step(self, ber: float) -> float:
        """
        Update weights using the observed BER and return the next hoprate.

        The BER is compared against the *previous* step's BER via the derivative
        metric.  On the very first call (no previous BER exists) the weights stay
        uniform and only a new hoprate is selected.

        Parameters
        ----------
        ber : float
            Mean BER observed during the environment step that just finished.

        Returns
        -------
        float
            The hoprate to use for the next environment step.
        """
        self._step_count += 1

        # Guard: first call has no prior BER to compare
        if self._previous_ber is None:
            self._previous_ber = float(ber)
            return self._select_hoprate()

        # Already saved prev BER / idx; now update weights
        ber_prev = self._previous_ber
        self._previous_ber = float(ber)

        self._update_weights(float(ber), ber_prev)
        self._normalize()
        return self._select_hoprate()

    def get_best_hoprate(self) -> float:
        """Return the hoprate with the highest current weight (MAP estimate)."""
        return self.candidates[np.argmax(self.weights)]

    def get_weighted_average(self) -> float:
        """Return the weighted-average hoprate (soft estimate)."""
        return float(np.average(self.candidates, weights=self.weights))

    def is_converged(self) -> bool:
        """Return True when max(weight) ≥ 1 − δ."""
        return bool(np.max(self.weights) >= 1.0 - self.delta)

    def get_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(candidates, weights)`` for diagnostics / plotting."""
        return self.candidates.copy(), self.weights.copy()

    def get_last_derivative(self) -> Optional[float]:
        """Return the derivative value from the last weight update."""
        return self._last_derivative

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_hoprate(self) -> Optional[float]:
        return self._current_hoprate

    @property
    def current_idx(self) -> Optional[int]:
        return self._current_idx

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Internal: query selection  (paper Algorithm 3.1)
    # ------------------------------------------------------------------

    def _select_hoprate(self) -> float:
        """
        Weighted-median query selection with randomisation (paper Algorithm 3.1).

        1. Find index *k* s.t. cumulative weight crosses W/2.
        2. Randomise between *k* and *k+1* with probability α proportional to
           the weight imbalance around the median.
        """
        total = np.sum(self.weights)
        cumulative = np.cumsum(self.weights)

        # weighted median index
        k = int(np.searchsorted(cumulative, total / 2.0))
        k = min(k, self.n_candidates - 1)

        if k < self.n_candidates - 1 and self.weights[k] > 0:
            sum_left = cumulative[k] - self.weights[k]   # Σ₁ᵏ⁻¹
            sum_right = total - cumulative[k]             # Σₖ₊₁ⁿ

            # α = (Σ₁ᵏ − Σₖ₊₁ⁿ) / (2 w_k)
            alpha = (sum_left + self.weights[k] - sum_right) / (2.0 * self.weights[k])
            alpha = float(np.clip(alpha, 0.0, 1.0))

            chosen = k if self._rng.random() < alpha else k + 1
        else:
            chosen = k

        self._previous_idx = self._current_idx
        self._previous_hoprate = self._current_hoprate
        self._current_idx = int(chosen)
        self._current_hoprate = float(self.candidates[chosen])
        return self._current_hoprate

    # ------------------------------------------------------------------
    # Internal: MWU weight update — DERIVATIVE VERSION
    # ------------------------------------------------------------------

    def _update_weights(self, ber: float, ber_prev: float) -> None:
        """
        Apply the MWU weight update using the derivative metric.

        The current hoprate *h_curr* is the query element.  The directional
        answer is derived from the metric (ΔBER_percent / Δhoprate_clamped):

            delta_ber_percent = (ber_curr - ber_prev) * 100
            delta_hoprate_clamped = h_curr - h_prev if (h_curr - h_prev) != 0 else -0.01
            metric = delta_ber_percent / delta_hoprate_clamped

            metric > threshold   →  answer = LEFT   →  h ≤ h_curr compatible
            metric ≤ threshold   →  answer = RIGHT  →  h ≥ h_curr compatible

        If Δhoprate is exactly 0, it is clamped to -0.01 to avoid division by zero.

        Compatible weights are multiplied by 2(1−p), incompatible by 2p.
        The split point is always the current hoprate index.
        """
        if (self._previous_idx is None or
            self._current_idx is None or
            self._previous_hoprate is None or
            self._current_hoprate is None):
            return

        curr_idx = self._current_idx

        # Calculate delta hoprate, handle zero case
        delta_hoprate = self._current_hoprate - self._previous_hoprate
        self._last_delta_hoprate = delta_hoprate
        delta_hoprate_clamped = delta_hoprate if delta_hoprate != 0 else -0.01

        # Calculate delta BER in percentage points (remove % sign)
        delta_ber = ber - ber_prev
        self._last_delta_ber = delta_ber
        delta_ber_percent = delta_ber * 100

        # Calculate metric
        metric = delta_ber_percent / delta_hoprate_clamped
        self._last_derivative = metric

        indices = np.arange(self.n_candidates)

        # Decision based on threshold
        if metric > self.derivative_threshold:
            # metric > threshold → answer = LEFT
            # → left side favoured
            favoured = indices < curr_idx
        else:
            # metric ≤ threshold → answer = RIGHT
            # → current + right side favoured
            favoured = indices >= curr_idx

        n_fav = np.sum(favoured)
        if n_fav == 0 or n_fav == self.n_candidates:
            return

        self.weights[favoured] *= 2.0 * (1.0 - self.p)
        self.weights[~favoured] *= 2.0 * self.p

    def _normalize(self) -> None:
        """Renormalise weights to sum to 1."""
        s = np.sum(self.weights)
        if s > 0:
            self.weights /= s
        else:
            # Degenerate fallback — reset to uniform
            self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    def _sim_ber(hoprate: float, optimal: float, max_hr: float,
                 baseline: float = 0.1, scale: float = 0.3,
                 rng: np.random.RandomState = None) -> float:
        """Simulated noisy BER: lower near *optimal* hoprate."""
        dist = abs(hoprate - optimal) / max_hr
        noise = rng.normal(0, 0.02) if rng else 0.0
        return max(0.0, min(1.0, baseline + scale * dist + noise))

    def run_test(label: str, optimal: float, steps: int = 80, threshold: float = -0.002):
        nbs = DerivativeNoisyBinarySearch(
            hoprate_min=10.0, hoprate_max=1000.0, hoprate_step=10.0,
            p=0.1, delta=0.05,
            derivative_threshold=threshold,
            seed=42,
        )
        print(f"\n{'='*60}")
        print(f"Mode: {label}")
        print(f"True optimal hoprate: {optimal:.0f} Hz")
        print(f"Derivative threshold: {threshold}")
        print(f"{'='*60}")

        h = nbs.reset()
        print(f"Step {nbs.step_count:3d}  init hoprate = {h:6.0f} Hz")

        for i in range(steps):
            ber = _sim_ber(nbs.current_hoprate, optimal, nbs.hoprate_max,
                           rng=nbs._rng)
            h = nbs.step(ber)

            metric = nbs.get_last_derivative()
            metric_str = f"{metric:10.4f}" if metric is not None else "      N/A"

            if nbs.is_converged():
                best = nbs.get_best_hoprate()
                wavg = nbs.get_weighted_average()
                print(f"Step {nbs.step_count:3d}  CONVERGED  "
                      f"hop={h:6.0f}  best={best:6.0f}  wavg={wavg:6.0f}  "
                      f"metric={metric_str}  max_w={np.max(nbs.weights):.4f}")
                print(f"  → estimate = {best:.0f} Hz  "
                      f"(error = {abs(best - optimal):.0f} Hz,  "
                      f"steps = {nbs.step_count})")
                return nbs

            if i % 10 == 0 or i == steps - 1:
                best = nbs.get_best_hoprate()
                wavg = nbs.get_weighted_average()
                max_w = np.max(nbs.weights)
                print(f"Step {nbs.step_count:3d}  hop={h:6.0f}  BER={ber:.4f}  "
                      f"best={best:6.0f}  wavg={wavg:6.0f}  "
                      f"metric={metric_str}  max_w={max_w:.4f}")

        best = nbs.get_best_hoprate()
        wavg = nbs.get_weighted_average()
        print(f"Did not converge in {steps} steps.")
        print(f"  best={best:.0f} Hz  wavg={wavg:.0f} Hz  "
              f"error={abs(best - optimal):.0f} Hz")
        return nbs

    # Test derivative-based MWU mapping
    run_test("Derivative-based (ΔBER%/Δhoprate ≤ threshold → RIGHT)",
             optimal=350.0, threshold=-0.002)

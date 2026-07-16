"""Generate and validate comb-jammer PSDs using the real RF configuration."""

import argparse
import os

import numpy as np

from fh_env_test import compute_psd_waterfall, save_psd_waterfall_plot
from jammers_test import FastNoiseSource, IndiscriminateJammer
import test_settings as settings


NUM_CHANNELS = 20
MIN_JAM_TO_CLEAR_MARGIN_DB = 3.0


def _channel_powers(waterfall_db, bins_per_channel):
    linear_psd = np.mean(10.0 ** (waterfall_db / 10.0), axis=0)
    return np.asarray(
        [
            np.sum(
                linear_psd[
                    channel * bins_per_channel:
                    (channel + 1) * bins_per_channel
                ]
            )
            for channel in range(NUM_CHANNELS)
        ]
    )


def validate_psd(output_dir=None):
    env_config = settings.ENV_CONFIG
    comb_config = settings.JAMMER_CONFIG["comb"]
    sweep_config = settings.JAMMER_CONFIG["sweep"]

    channel_width = float(env_config["Sub_interval"])
    comb_bandwidth = float(comb_config["bandwidth"])
    comb_spacing = float(comb_config["sub_interval"])
    if not np.isclose(channel_width, comb_bandwidth) or not np.isclose(
        channel_width, comb_spacing
    ):
        raise AssertionError(
            "Channel width, comb bandwidth, and comb spacing must match."
        )

    fs = float(env_config["Fs"])
    start_frequency = float(env_config["Startfre"])
    end_frequency = float(env_config["Endfre"])
    duration = 0.1
    df = 10000.0
    bins_per_channel = int(round(channel_width / df))
    if bins_per_channel <= 0 or not np.isclose(
        bins_per_channel * df, channel_width
    ):
        raise AssertionError("PSD bins must divide each channel exactly.")

    np.random.seed(settings.RANDOM_SEED)

    # A 110 ms source is sufficient for this 100 ms check and avoids allocating
    # the full one-second noise cache used by a training environment.
    noise_source = FastNoiseSource(
        fs, comb_bandwidth, duration=duration + 0.01
    )
    jammer = IndiscriminateJammer(
        Fs=fs,
        sweep_config=sweep_config,
        comb_config=comb_config,
        noise_source=noise_source,
        mode="comb",
    )

    if output_dir is None:
        output_dir = os.path.join(settings.DEFAULT_OUTPUT_DIR, "psd_check")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(settings.RANDOM_SEED)
    t = np.arange(int(duration * fs), dtype=np.float64) / fs
    all_channels = set(range(NUM_CHANNELS))
    report = []

    for phase in (0, 1):
        jammer.comb_phase = phase
        jammer_signal, _ = jammer.generate(
            t, start_frequency, end_frequency
        )
        observed_signal = jammer_signal + float(
            env_config["noise_std"]
        ) * rng.standard_normal(len(jammer_signal))

        waterfall = compute_psd_waterfall(
            observed_signal,
            fs=fs,
            f_start=start_frequency,
            f_end=end_frequency,
            dt=0.001,
            df=df,
            max_duration=duration,
            plot=False,
        )
        save_psd_waterfall_plot(
            waterfall,
            os.path.join(output_dir, f"phase_{phase}.png"),
            plot_title=f"Comb Phase {phase} PSD Check",
        )

        channel_powers = _channel_powers(waterfall, bins_per_channel)
        expected = set(settings.COMB_PHASE_CHANNELS[phase])
        strongest = set(
            np.argsort(channel_powers)[-len(expected):].tolist()
        )
        if strongest != expected:
            raise AssertionError(
                f"Phase {phase}: strongest channels {sorted(strongest)} do not "
                f"match configured comb channels {sorted(expected)}."
            )

        clear = all_channels - expected
        margin_db = 10.0 * np.log10(
            np.min(channel_powers[list(expected)])
            / np.max(channel_powers[list(clear)])
        )
        if margin_db < MIN_JAM_TO_CLEAR_MARGIN_DB:
            raise AssertionError(
                f"Phase {phase}: jam-to-clear PSD margin is only "
                f"{margin_db:.2f} dB."
            )
        report.append((phase, sorted(strongest), float(margin_db)))

    return report, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the two comb phases in the PSD domain."
    )
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validation_report, saved_dir = validate_psd(args.output_dir)
    print(
        "Channel width = comb bandwidth = "
        f"{settings.CHANNEL_WIDTH:.0f} Hz"
    )
    for phase, strongest_channels, margin_db in validation_report:
        print(
            f"Phase {phase}: strongest channels = {strongest_channels}, "
            f"jam-to-clear margin = {margin_db:.2f} dB"
        )
    print(f"PSD validation images saved to {saved_dir}")

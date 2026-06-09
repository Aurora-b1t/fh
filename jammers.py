import numpy as np

# -----------------------------
# 快速噪声源
# -----------------------------
class FastNoiseSource:
    """
    预生成长段带限噪声的缓冲区，避免反复计算 FFT/IFFT。
    """
    def __init__(self, Fs, bandwidth, duration=1.0):
        self.length = int(Fs * duration)
        # 1. Generate Gaussian noise
        n = np.random.randn(self.length).astype(np.float32)
        # 2. FFT
        spec = np.fft.rfft(n)
        # 3. Low-pass
        if bandwidth > 0:
            cutoff_idx = int(np.floor(bandwidth * self.length / (2.0 * Fs)))
            if cutoff_idx + 1 < len(spec):
                spec[cutoff_idx + 1:] = 0
                
        # 4. IFFT
        n_lp = np.fft.irfft(spec, n=self.length)
        # 5. Normalize
        std_val = np.std(n_lp)
        if std_val > 1e-12:
            n_lp /= std_val
        self.noise = n_lp

    def get_noise(self, num_samples):
        if num_samples >= self.length:
            # 这种情况下直接返回全部并循环填充
            tile_count = (num_samples // self.length) + 1
            return np.tile(self.noise, tile_count)[:num_samples]
            
        start = np.random.randint(0, self.length - num_samples)
        return self.noise[start : start + num_samples]


# -----------------------------
# 干扰机
# -----------------------------
class ReactiveJammer:
    def __init__(self, Fs, speed=10.0, power=0.5, bandwidth=50000.0, noise_source=None):
        self.Fs = float(Fs)
        self.speed = float(speed)
        self.power = float(power)
        self.bandwidth = float(bandwidth)
        # Shared noise source
        if noise_source is None:
            self.noise_source = FastNoiseSource(Fs, bandwidth)
        else:
            self.noise_source = noise_source

    def generate(self, t, hop_seq, Startfre, Sub_interval, hoprate):
        N = len(t)
        if hoprate > self.speed or N == 0:
            return np.zeros_like(t), False

        # Use fast noise source
        raw_noise = self.noise_source.get_noise(N)
        baseband_noise = raw_noise * self.power

        s_per_hop = self.Fs / float(hoprate)
        jam = np.zeros(N)
        pos = 0
        k = 0
        
        phase_k = 2 * np.pi / self.Fs

        while pos < N and k < len(hop_seq):
            next_pos = int(round((k + 1) * s_per_hop))
            next_pos = min(N, max(pos + 1, next_pos))
            length = next_pos - pos
            
            f_c = Startfre + hop_seq[k] * Sub_interval + 0.5 * Sub_interval
            
            t_seg_local = np.arange(length) 
            carrier = np.cos((phase_k * f_c) * t_seg_local)
            
            jam[pos:next_pos] = baseband_noise[pos:next_pos] * carrier
            
            pos = next_pos
            k += 1
            
        if pos < N:
            last_idx = max(0, min(k - 1, len(hop_seq) - 1))
            f_last = Startfre + hop_seq[last_idx] * Sub_interval + 0.5 * Sub_interval
            length = N - pos
            t_seg_local = np.arange(length)
            carrier = np.cos((phase_k * f_last) * t_seg_local)
            jam[pos:] = baseband_noise[pos:] * carrier
            
        return jam, True


class IndiscriminateJammer:
    def __init__(self, Fs, sweep_config=None, comb_config=None, 
                 noise_source=None, mode='sweep'):
        self.Fs = float(Fs)
        self.mode = mode
        
        # Default Configs
        self.sweep_config = sweep_config if sweep_config else {}
        self.comb_config = comb_config if comb_config else {}
        
        # --- Sweep Params ---
        self.s_step = float(self.sweep_config.get('step', 125000.0))
        self.s_power = float(self.sweep_config.get('power', 0.8))
        self.s_dwell = float(self.sweep_config.get('dwell_time', 0.004))
        self.s_bw = float(self.sweep_config.get('bandwidth', 30000.0))
        
        # --- Comb Params ---
        self.c_step = float(self.comb_config.get('step', 100000.0))
        self.c_power = float(self.comb_config.get('power', 0.5))
        self.c_bw = float(self.comb_config.get('bandwidth', 30000.0))
        
        self._sweep_idx = 0
        
        # --- Noise Sources ---
        if noise_source:
             self.ns_sweep = noise_source
             self.ns_comb = noise_source
        else:
             self.ns_sweep = FastNoiseSource(Fs, self.s_bw)
             # Reuse if bandwidth same
             if self.c_bw == self.s_bw:
                 self.ns_comb = self.ns_sweep
             else:
                 self.ns_comb = FastNoiseSource(Fs, self.c_bw)

        # State for changing Comb frequencies
        self.comb_phase = 0  # 0 or 1
        
        # Pre-computed buffers
        self.pre_buffer_sweep = None
        self.pre_buffer_comb0 = None
        self.pre_buffer_comb1 = None
        self.pre_len = 0
        self.pre_ptr = 0 # Not used internally, caller must manage or we use internal ptr?
                         # Usually caller manages because reset() logic is in Env.

    def set_mode(self, mode):
        if mode in ['sweep', 'comb', 'both']:
            self.mode = mode

    def step_comb(self):
        """Toggle the comb jamming frequency group."""
        self.comb_phase = 1 - self.comb_phase

    def reset_comb(self):
        self.comb_phase = 0

    def precompute(self, duration=4.4, Startfre=3e6, Endfre=4e6):
        """
        Pre-compute sweep and comb jamming signals for a fixed duration.
        """
        print(f"Pre-computing Jammers (Duration: {duration}s)...")
        num_samples = int(duration * self.Fs)
        self.pre_len = num_samples
        t = np.arange(num_samples) / self.Fs # Absolute time for continuity?
        # Actually generate() uses t just for length usually, BUT carrier phase
        # continuity across calls requires care.
        # Here we generate one long coherent block.
        
        # 1. Sweep Buffer
        # Temporarily save mode/state
        old_mode = self.mode
        old_idx = self._sweep_idx
        
        self.mode = 'sweep'
        self._sweep_idx = 0 # Start from beginning of sweep cycle
        # Note: generate() expects t relative to 0 for carrier phase if we call it naively.
        # But we want carrier phase continuous across the whole duration.
        # My current generate() implementation uses t_seg_local = np.arange(length) inside the loop,
        # resets phase at every dwell segment? No.
        # `carrier = np.cos((phase_k * f) * t_seg_local)`
        # Yes, it resets phase at every segment boundary in `generate`. 
        # But since we call generate ONCE for the whole duration, it will be consistent within that duration.
        
        # However, generate() chunks by dwell time. 
        # Inside generate loop: `t_seg_local` starts at 0 for each dwell segment.
        # This implies phase discontinuity at frequency hops. This is physically acceptable for a jammer (oscillator retuning).
        
        jam_s, _ = self.generate(t, Startfre, Endfre)
        self.pre_buffer_sweep = jam_s.astype(np.float32)
        
        # 2. Comb Buffer 0
        self.mode = 'comb'
        self.comb_phase = 0
        jam_c0, _ = self.generate(t, Startfre, Endfre)
        self.pre_buffer_comb0 = jam_c0.astype(np.float32)
        
        # 3. Comb Buffer 1
        self.mode = 'comb'
        self.comb_phase = 1
        jam_c1, _ = self.generate(t, Startfre, Endfre)
        self.pre_buffer_comb1 = jam_c1.astype(np.float32)
        
        # Restore state
        self.mode = old_mode
        self._sweep_idx = old_idx
        print("Jammer Pre-computation complete.")

    def get_composite_signal(self, start_sample_idx, num_samples):
        """
        Retrieve a slice of the pre-computed jamming signal based on current mode and comb phase.
        Auto-wraps if exceeding buffer duration.
        """
        if self.pre_buffer_sweep is None:
            # Fallback (should not happen if precompute called)
            return np.zeros(num_samples)
            
        # Determine slices (handling wrap-around)
        idx_start = start_sample_idx % self.pre_len
        idx_end = idx_start + num_samples
        
        # Helper to get slice from a buffer
        def get_slice(buf):
            if idx_end <= self.pre_len:
                return buf[idx_start:idx_end]
            else:
                # Wrap around
                part1 = buf[idx_start:]
                rem = idx_end - self.pre_len
                part2 = buf[:rem]
                return np.concatenate([part1, part2])

        jam_total = np.zeros(num_samples, dtype=np.float32)
        
        if self.mode == 'sweep' or self.mode == 'both':
            jam_total += get_slice(self.pre_buffer_sweep)
            
        if self.mode == 'comb' or self.mode == 'both':
            if self.comb_phase == 0:
                jam_total += get_slice(self.pre_buffer_comb0)
            else:
                jam_total += get_slice(self.pre_buffer_comb1)
                
        return jam_total

    def generate(self, t, Startfre, Endfre):
        N = len(t)
        if N == 0:
            return np.zeros_like(t), []
            
        jam = np.zeros_like(t)
        freqs_used = []
        phase_k = 2 * np.pi / self.Fs
        bw_total = max(Endfre - Startfre, 1.0)

        # -----------------------------
        # Comb Jamming
        # -----------------------------
        if self.mode == 'comb' or self.mode == 'both':
            # Use Comb Noise Source
            raw_noise_c = self.ns_comb.get_noise(N)
            baseband_noise_c = raw_noise_c * self.c_power
            
            # --- Optimized for 8 fixed points aligned to 50kHz channels ---
            sub_interval = 50000.0
            
            # Hardcoded selection for two alternating 8-channel comb groups.
            if self.comb_phase == 0:
                # Group 0: even channel indices
                target_indices = np.array([0, 2, 4, 6, 8, 10, 12, 14])
            else:
                # Group 1: odd channel indices
                target_indices = np.array([1, 3, 5, 7, 9, 11, 13, 15])
            
            # Map to frequencies: Start + k*sub + 0.5*sub
            freqs = Startfre + target_indices * sub_interval + 0.5 * sub_interval
            freqs = freqs[(freqs >= Startfre) & (freqs < Endfre)]
            
            if len(freqs) > 0:
                combined_carrier = np.zeros(N)
                t_local = np.arange(N)
                
                # Normalize power so total power remains roughly consistent or per-tone?
                # Usually comb power is defined per tone or total? 
                # Code originally: norm_factor = 1.0 / sqrt(len). 
                # This keeps TOTAL power = baseband power.
                norm_factor = 1.0 / np.sqrt(len(freqs))
                
                for f in freqs:
                    combined_carrier += np.cos((phase_k * f) * t_local)
                    freqs_used.append(f)
                
                combined_carrier *= norm_factor
                jam += baseband_noise_c * combined_carrier

        # -----------------------------
        # Sweep Jamming
        # -----------------------------
        if self.mode == 'sweep' or self.mode == 'both':
            # Use Sweep Noise Source
            raw_noise_s = self.ns_sweep.get_noise(N)
            baseband_noise_s = raw_noise_s * self.s_power
            
            # Sweep Params
            step = self.s_step
            dwell = self.s_dwell
            
            samples_per_dwell = max(1, int(round(dwell * self.Fs)))
            num_segments = int(np.ceil(len(t) / samples_per_dwell))
            num_steps = max(1, int(np.floor(bw_total / step)))
            
            idx = self._sweep_idx
            
            for seg in range(num_segments):
                f = Startfre + (idx % num_steps) * step + step / 2.0
                if f >= Endfre:
                    f = Startfre + np.mod(f - Startfre, bw_total)
                
                s = seg * samples_per_dwell
                e = min(len(t), (seg + 1) * samples_per_dwell)
                length = e - s
                
                t_seg_local = np.arange(length)
                carrier = np.cos((phase_k * f) * t_seg_local)
                jam[s:e] += baseband_noise_s[s:e] * carrier
                
                freqs_used.append(f)
                idx += 1
            self._sweep_idx = idx

        return jam, freqs_used

import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from scipy.signal import upfirdn
import gymnasium as gym
from gymnasium import spaces
import time
from jammers import FastNoiseSource, ReactiveJammer, IndiscriminateJammer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 基础函数
# -----------------------------
def rcosdesign_srv(rolloff, span, sps):
    """
    Root raised cosine filter normalized to unit energy.
    """
    # Ensure odd length (span*sps + 1) so that group delay is integer and aligned with sps
    rrc_filter = rrcosfilter(span * sps + 1, rolloff, 1, sps)[1]
    rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter ** 2) + 1e-12)
    return rrc_filter


def compute_psd_waterfall(signal, fs, f_start, f_end,
                          dt=0.001,      # 1 ms time resolution
                          df=10000.0,    # 10 kHz frequency resolution
                          max_duration=0.1,  # analyze first 100 ms
                          window='hann',
                          plot=False,
                          plot_title=""):
    """
    Optimized Compute PSD waterfall.
    """
    # ---- trim to first max_duration seconds ----
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * fs)
        signal = signal[:max_samples]

    Nwin = int(dt * fs)
    if Nwin <= 0:
        raise ValueError("Nwin must be positive. Check dt and fs.")
    if len(signal) < Nwin:
        return np.zeros((0, max(1, int(np.floor((f_end - f_start) / df)))) )

    step = Nwin

    if window == 'hann':
        win = np.hanning(Nwin)
    elif window == 'hamming':
        win = np.hamming(Nwin)
    else:
        win = np.ones(Nwin)
        
    win_sq_sum = np.sum(win ** 2)

    Nfft = int(2 ** np.ceil(np.log2(Nwin)))
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    n_bins = int(np.floor((f_end - f_start) / df))
    n_bins = max(1, n_bins)
    f_bin_edges = f_start + np.arange(n_bins + 1) * df

    # Vectorized bin search
    # Find insertion points of edges in freqs
    # freqs is sorted.
    edge_indices = np.searchsorted(freqs, f_bin_edges)
    
    valid_bins = []
    for k in range(n_bins):
        start_idx = edge_indices[k]
        end_idx = edge_indices[k+1]
        if end_idx > start_idx:
            valid_bins.append((k, start_idx, end_idx))

    waterfall = []
    num_frames = (len(signal) - Nwin) // step + 1
    
    # Pre-calculate constants
    norm_factor = 1.0 / (fs * win_sq_sum / Nwin)

    for i in range(num_frames):
        start = i * step
        frame = signal[start:start + Nwin]
        if len(frame) < Nwin:
            break

        frame_win = frame * win
        spec = np.fft.rfft(frame_win, n=Nfft)
        # Compute PSD directly
        psd = (np.abs(spec) ** 2) * norm_factor

        band_powers = np.zeros(n_bins)
        
        # Optimize loop using slicing
        for k, s_idx, e_idx in valid_bins:
            # sum is much faster on slice than np.where
            band_powers[k] = np.sum(psd[s_idx:e_idx])

        waterfall.append(band_powers)

    waterfall = np.array(waterfall)  # [time, freq_bin]
    eps = 1e-12
    waterfall_db = 10 * np.log10(waterfall + eps)

    if plot and waterfall_db.size > 0:
        plt.figure(figsize=(8, 4))
        plt.imshow(waterfall_db.T, origin="lower", aspect="auto", cmap="jet")
        plt.colorbar(label='PSD (dB)')
        plt.xlabel('Time bin')
        plt.ylabel('Freq bin')
        title_str = 'PSD Waterfall ({:.0f} ms)'.format(max_duration * 1e3)
        if plot_title:
            title_str += f"\n{plot_title}"
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    return waterfall_db

# -----------------------------
# M序列（LFSR）生成
# -----------------------------
def generate_mseq_states(n_bits=10, length=1000, taps=(10, 7), seed=1):
    if seed == 0:
        seed = 1
    mask = (1 << n_bits) - 1
    state = seed & mask
    seq = []
    for _ in range(length):
        seq.append(state)
        fb = 0
        for t in taps:
            fb ^= (state >> (n_bits - t)) & 1
        state = ((state << 1) & mask) | fb
        if state == 0:
            state = 1
    return np.array(seq, dtype=np.int64)




# -----------------------------
# 调制与信道
# -----------------------------
class QPSKModem:
    def __init__(self, Baud, Fs, Ns, Nh):
        self.Baud = Baud
        self.Fs = Fs
        self.Ns = Ns
        self.Nh = Nh
        self.LBF = rcosdesign_srv(0.5, 16, Ns)

    def generate_bits(self, Bitrate):
        return np.random.binomial(n=1, p=0.5, size=Bitrate)

    def pulse_shape(self, bits):
        bits = np.asarray(bits).astype(np.int8)
        if len(bits) % 2 != 0:
            bits = np.concatenate([bits, np.array([0], dtype=np.int8)])

        I_bits = 2 * bits[0::2] - 1
        Q_bits = 2 * bits[1::2] - 1
        n_syms = len(I_bits)

        I_f = upfirdn(self.LBF, I_bits, up=self.Ns)
        Q_f = upfirdn(self.LBF, Q_bits, up=self.Ns)
        gd = (len(self.LBF) - 1) // 2
        I_pulse = I_f[gd:gd + n_syms * self.Ns]
        Q_pulse = Q_f[gd:gd + n_syms * self.Ns]
        return I_pulse, Q_pulse


class FHSSChannel:
    def __init__(self, Startfre, Sub_interval, Hoprate, Fs):
        self.Startfre = Startfre
        self.Sub_interval = Sub_interval
        self.Hoprate = Hoprate
        self.Fs = Fs

    def hop_carrier(self, t, hop_seq):
        """
        Return complex carrier directly: exp(j*phi)
        """
        N = len(t)
        if N == 0:
            return np.zeros(0, dtype=complex)

        s_per_hop = self.Fs / float(self.Hoprate)
        hop_idx = np.empty(N, dtype=int)
        pos = 0
        k = 0
        while pos < N and k < len(hop_seq):
            next_pos = int(round((k + 1) * s_per_hop))
            next_pos = min(N, max(pos + 1, next_pos))
            hop_idx[pos:next_pos] = hop_seq[k]
            pos = next_pos
            k += 1
        if pos < N:
            last_idx = max(0, min(k - 1, len(hop_seq) - 1))
            hop_idx[pos:] = hop_seq[last_idx]

        hop_fre = self.Startfre + hop_idx * self.Sub_interval + 0.5 * self.Sub_interval
        phase = 2 * np.pi * np.cumsum(hop_fre) / self.Fs
        # Optimized: calculate complex exponential directly
        carrier_complex = np.exp(1j * phase)
        return carrier_complex

    def transmit(self, I_pulse, Q_pulse, carrier_complex, noise_std=0.1):
        # carrier_complex is exp(j*phi)
        baseband = I_pulse + 1j * Q_pulse
        rf_complex = baseband * carrier_complex
        noise = noise_std * np.random.randn(len(rf_complex))
        return rf_complex, noise


class QPSKReceiver:
    def __init__(self, Ns):
        self.Ns = Ns
        self.MF = rcosdesign_srv(0.5, 16, Ns)

    def demodulate(self, modu_signal, bits, carrier_complex):
        # demodulate multiplying by conj(carrier)
        demod_complex = modu_signal * np.conj(carrier_complex)

        y_complex = upfirdn(self.MF, demod_complex, up=1, down=self.Ns)

        gd = (len(self.MF) - 1) // 2
        start_idx = gd // self.Ns
        
        y_complex = y_complex[start_idx:]

        y_i_end = np.real(y_complex)
        y_q_end = np.imag(y_complex)

        y_i_end = (y_i_end >= 0).astype(int)
        y_q_end = (y_q_end >= 0).astype(int)

        num_syms = len(bits) // 2
        min_len = min(len(y_i_end), len(y_q_end), num_syms)
        y_i_end = y_i_end[:min_len]
        y_q_end = y_q_end[:min_len]

        receive_data = np.zeros(2 * min_len, dtype=int)
        receive_data[::2] = y_i_end
        receive_data[1::2] = y_q_end

        bit_error = np.mean(receive_data != bits[:2 * min_len]) if min_len > 0 else 0.0
        return receive_data, bit_error





# -----------------------------
# Gym 环境封装
# -----------------------------
class FHSSQPSKEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 Startfre=3e6,
                 Endfre=4e6,
                 Fs=1e7,
                 Sub_interval=50000,
                 Hoprate=100,
                 hoprate_min=10.0,
                 hoprate_max=1000.0,
                 Baud=25000,
                 dt=0.001,
                 df=10000.0,
                 enable_reactive=False,
                 reactive_speed=160.0,
                 reactive_power=0.5,
                 reactive_bandwidth=50000.0,
                 enable_sweep=True,
                 sweep_step=200000,
                 sweep_power=0.5,
                 sweep_dwell=0.004,
                 sweep_bandwidth=50000.0,
                 sweep_mode='comb',
                 enable_rayleigh=False,
                 rayleigh_coherence=800,
                 mseq_length=1000,
                 mseq_nbits=10,
                 mseq_taps=(10, 7),
                 mseq_seed=1,
                 debug_plot_psd=False,
                 debug_log_hops=False,
                 reset_mseq_each_step=True):
        super().__init__()

        self.Startfre = float(Startfre)
        self.Endfre = float(Endfre)
        self.Fs = int(Fs)
        self.Sub_interval = float(Sub_interval)
        self.Baud = int(Baud)
        self.Tb = 1.0 / self.Baud
        self.Ns = int(self.Fs / self.Baud)
        self.dt = float(dt)
        self.df = float(df)

        self.num_channels = int(round((self.Endfre - self.Startfre) / self.Sub_interval))
        self.num_channels = max(1, self.num_channels)

        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max) if hoprate_max is not None else float(Baud)
        self.base_hoprate = float(Hoprate)

        self.Nh = 200 if (200 % 2 == 0) else 202
        self.current_hoprate = float(Hoprate)
        self.modem = QPSKModem(self.Baud, self.Fs, self.Ns, self.Nh)
        self.channel = FHSSChannel(self.Startfre, self.Sub_interval, self.current_hoprate, self.Fs)
        self.receiver = QPSKReceiver(self.Ns)
        
        self.enable_reactive = bool(enable_reactive)
        self.enable_sweep = bool(enable_sweep)

        # Share noise source if bandwidths match
        shared_noise_source = None
        if self.enable_reactive and self.enable_sweep and reactive_bandwidth == sweep_bandwidth:
            shared_noise_source = FastNoiseSource(self.Fs, reactive_bandwidth)
        
        if self.enable_reactive:
            ns = shared_noise_source if shared_noise_source else None
            # If no shared source (bandwidth mismatch or only one enabled), ReactiveJammer will create one if ns is None
            # But wait, ReactiveJammer default behavior is creating new one.
            # If shared_noise_source is None but we haven't created one, we let it create.
            self.reactive = ReactiveJammer(Fs=self.Fs,
                                           speed=reactive_speed,
                                           power=reactive_power,
                                           bandwidth=reactive_bandwidth,
                                           noise_source=ns)
        else:
            self.reactive = None
                                       
        if self.enable_sweep:
            ns = shared_noise_source if shared_noise_source else None
            # If shared_noise_source is used by reactive, we reuse it.
            self.sweep = IndiscriminateJammer(Fs=self.Fs,
                                              step=sweep_step if sweep_step is not None else self.Sub_interval,
                                              power=sweep_power,
                                              dwell_time=sweep_dwell,
                                              bandwidth=sweep_bandwidth,
                                              noise_source=ns,
                                              mode=sweep_mode)
        else:
            self.sweep = None

        self.enable_rayleigh = bool(enable_rayleigh)
        self.rayleigh_coherence = float(rayleigh_coherence)

        self.mseq_states = generate_mseq_states(n_bits=mseq_nbits,
                                                length=mseq_length,
                                                taps=mseq_taps,
                                                seed=mseq_seed)
        self.mseq_channels = (self.mseq_states % self.num_channels).astype(int)
        self._mseq_ptr = 0
        self.reset_mseq_each_step = bool(reset_mseq_each_step)

        self.action_space = spaces.Dict({
            "hoprate": spaces.Box(
                low=np.array([self.hoprate_min], dtype=np.float32),
                high=np.array([self.hoprate_max], dtype=np.float32),
                dtype=np.float32
            ),
            "offsets": spaces.Box(
                low=np.zeros(10, dtype=np.float32),
                high=np.full(10, max(1, self.num_channels - 1), dtype=np.float32),
                shape=(10,),
                dtype=np.float32
            ),
        })

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32
        )

        self.state = None
        self.last_info = {}

        self.debug_plot_psd = bool(debug_plot_psd)
        self.debug_log_hops = bool(debug_log_hops)
        self.current_step = 0

        self._apply_hoprate(self.base_hoprate)

    def seed(self, seed=None):
        np.random.seed(seed)

    def _apply_hoprate(self, hoprate_target):
        hoprate_clip = float(np.clip(hoprate_target, self.hoprate_min, self.hoprate_max))
        hoprate_used = float(int(round(hoprate_clip / 10.0)) * 10)

        Nh = max(2, int(round(self.Baud / max(hoprate_used, 1e-9))))
        if Nh % 2 != 0:
            Nh += 1

        self.Nh = Nh
        self.modem.Nh = Nh
        self.current_hoprate = hoprate_used
        self.channel.Hoprate = hoprate_used

        return {
            "hoprate_action": hoprate_target,
            "hoprate_used": hoprate_used,
            "Nh_used": Nh
        }

    def _generate_rayleigh(self, length):
        if (not self.enable_rayleigh) or length <= 0:
            return None
        coh = max(1, int(self.rayleigh_coherence))
        num_seg = int(np.ceil(length / coh))
        mags = np.random.rayleigh(scale=np.sqrt(2) / 2, size=num_seg)
        mag_seq = np.repeat(mags, coh)[:length]
        return mag_seq

    def _observe_100ms(self, block_id=None):
        N_obs = int(0.1 * self.Fs)
        t_obs = np.arange(N_obs) / self.Fs

        sweep_jam = np.zeros(N_obs)
        if self.enable_sweep and self.sweep is not None:
            sweep_jam, _ = self.sweep.generate(t_obs, self.Startfre, self.Endfre)

        noise = 0.1 * np.random.randn(N_obs)
        obs_signal = sweep_jam + noise

        plot_title = ""
        do_plot = self.debug_plot_psd
        if block_id is not None:
            plot_title = f"Step {self.current_step} - Block {block_id}"

        waterfall_db = compute_psd_waterfall(
            obs_signal,
            fs=self.Fs,
            f_start=self.Startfre,
            f_end=self.Endfre,
            dt=self.dt,
            df=self.df,
            max_duration=0.1,
            plot=do_plot,
            plot_title=plot_title
        )
        return waterfall_db

    def _get_block_hopseq(self, hops_per_block, offset):
        if hops_per_block <= 0:
            return np.array([], dtype=int)

        end_ptr = self._mseq_ptr + hops_per_block
        if end_ptr <= len(self.mseq_channels):
            base = self.mseq_channels[self._mseq_ptr:end_ptr]
        else:
            part1 = self.mseq_channels[self._mseq_ptr:]
            part2 = self.mseq_channels[:(end_ptr - len(self.mseq_channels))]
            base = np.concatenate([part1, part2])
        
        off_int = int(np.round(offset)) % self.num_channels
        hop_seq = (base + off_int) % self.num_channels
        return hop_seq

    def reset(self):
        self.current_step = 0
        _ = self._apply_hoprate(self.base_hoprate)

        obs = self._observe_100ms(block_id=0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.state = obs.astype(np.float32)
        self.last_info = {
            "ber_blocks": [],
            "hoprate_used": self.current_hoprate,
        }
        return self.state, self.last_info

    def step(self, action=None):
        self.current_step += 1
        if action is None:
            hoprate_action = self.base_hoprate
            offsets_action = np.zeros(10, dtype=np.int32)
        else:
            if isinstance(action, dict):
                hoprate_action = float(action.get("hoprate", self.base_hoprate))
                offsets_action = np.array(action.get("offsets", np.zeros(10)), dtype=np.float32)
            elif isinstance(action, (list, tuple)) and len(action) == 2:
                hoprate_action = float(action[0])
                offsets_action = np.array(action[1], dtype=np.float32)
            else:
                hoprate_action = float(action)
                offsets_action = np.zeros(10, dtype=np.float32)

        offsets_action = np.array(offsets_action, dtype=np.float32)
        if offsets_action.shape != (10,):
            raise ValueError("offsets 必须是长度为10的向量。")

        ainfo = self._apply_hoprate(hoprate_action)

        hops_per_block = int(round(self.current_hoprate * 0.1))
        hops_per_block = max(1, hops_per_block)

        ber_blocks = []
        reactive_active_blocks = []

        for b in range(10):
            num_syms_block = int(round(self.Baud * 0.1))
            bits_block = self.modem.generate_bits(2 * num_syms_block)
            I_pulse, Q_pulse = self.modem.pulse_shape(bits_block)

            
            t_block = np.arange(len(I_pulse)) / self.Fs

            hop_seq_block = self._get_block_hopseq(hops_per_block, offsets_action[b])
            self._mseq_ptr = (self._mseq_ptr + len(hop_seq_block)) % len(self.mseq_channels)

            carrier_complex = self.channel.hop_carrier(t_block, hop_seq_block)
            
            rf_complex, noise = self.channel.transmit(I_pulse, Q_pulse, carrier_complex, noise_std=0.1)
            
            rayleigh_mag = self._generate_rayleigh(len(I_pulse))
            if rayleigh_mag is None:
                rayleigh_mag = np.ones_like(I_pulse)

            reactive_jam = np.zeros(len(I_pulse))
            reactive_active = False
            if self.enable_reactive and self.reactive is not None:
                reactive_jam, reactive_active = self.reactive.generate(
                    t_block, hop_seq_block, self.Startfre, self.Sub_interval, self.current_hoprate
                )
            sweep_jam = np.zeros(len(I_pulse))
            if self.enable_sweep and self.sweep is not None:
                sweep_jam, _ = self.sweep.generate(t_block, self.Startfre, self.Endfre)
            
            rx_real = np.real(rf_complex * rayleigh_mag) + reactive_jam + sweep_jam + noise
            #start_time = time.time()

            _, ber = self.receiver.demodulate(rx_real, bits_block, carrier_complex)
            ber_blocks.append(float(ber))
            reactive_active_blocks.append(bool(reactive_active))

            #end_time = time.time()
            #print(f"Block {b+1} processing time: {end_time - start_time:.4f} s")


            if self.debug_plot_psd:
                compute_psd_waterfall(
                    rx_real,
                    fs=self.Fs,
                    f_start=self.Startfre,
                    f_end=self.Endfre,
                    dt=self.dt,
                    df=self.df,
                    max_duration=0.1,
                    plot=True,
                    plot_title=f"Step {self.current_step} - Block {b+1}"
                )
            if self.debug_log_hops:
                print(f"Block {b+1}: Hop Seq (with offset) = {hop_seq_block.tolist()}")

        obs = self._observe_100ms(block_id=0)
        self.state = obs.astype(np.float32)

        if self.reset_mseq_each_step:
            self._mseq_ptr = 0

        mean_ber = float(np.mean(ber_blocks)) if len(ber_blocks) > 0 else 0.0
        reward = 0.5 - mean_ber - ainfo["hoprate_used"] * 0.0001

        self.last_info = {
            "ber_blocks": ber_blocks,
            "mean_ber": mean_ber,
            "hoprate_used": ainfo["hoprate_used"],
            "hops_per_block": hops_per_block,
            "reactive_active_blocks": reactive_active_blocks,
        }

        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, self.last_info

    def render(self, mode="human"):
        if self.state is None:
            return
        plt.figure(figsize=(8, 4))
        plt.imshow(self.state.T, origin="lower", aspect="auto", cmap="jet")
        plt.colorbar(label='PSD (dB)')
        plt.xlabel('Time bin')
        plt.ylabel('Freq bin')
        plt.title('PSD Waterfall (100 ms observation)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = FHSSQPSKEnv(enable_reactive=True,
                      enable_sweep=True,
                      enable_rayleigh=True,
                      debug_plot_psd=False,
                      debug_log_hops=False)
    obs, info = env.reset()

    offsets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action = {"hoprate": 200.0, "offsets": offsets}
    for i in range(20):
        start_time = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        end_time = time.time()
        print(f"Step {i+1}: Reward: {reward}, Mean BER: {info['mean_ber']}, Step Time: {end_time - start_time:.4f} s")
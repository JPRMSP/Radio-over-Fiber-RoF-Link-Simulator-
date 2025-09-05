import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helper Functions
# -------------------------------

def generate_signal(mod_type, N=2048, fc=50e3, fs=1e6, M=16):
    t = np.arange(N) / fs

    if mod_type == "AM":
        msg = np.cos(2 * np.pi * 1e3 * t)
        carrier = np.cos(2 * np.pi * fc * t)
        signal = (1 + msg) * carrier
        data = None

    elif mod_type == "FM":
        msg = np.cos(2 * np.pi * 1e3 * t)
        signal = np.cos(2 * np.pi * fc * t + 5 * np.sin(2 * np.pi * 1e3 * t))
        data = None

    elif mod_type == "QAM":
        k = int(np.log2(M))  # bits per symbol
        symbols = (np.random.randint(0, int(np.sqrt(M)), N) - (np.sqrt(M)-1)/2) \
                + 1j*(np.random.randint(0, int(np.sqrt(M)), N) - (np.sqrt(M)-1)/2)
        symbols /= np.sqrt((np.mean(np.abs(symbols)**2)))  # normalize power
        data = symbols
        signal = np.real(symbols * np.exp(1j * 2 * np.pi * fc * t))

    else:
        signal = np.zeros(N)
        data = None

    return t, signal, data


def fiber_channel(signal, fiber_length_km, alpha=0.2, noise_power=0.01):
    attenuation = 10 ** (-alpha * fiber_length_km / 20)
    noisy_signal = attenuation * signal + np.sqrt(noise_power) * np.random.randn(len(signal))
    return noisy_signal


def plot_time_freq(t, signal, fs, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(t[:500], signal[:500])
    axs[0].set_title(f"{title} - Time Domain")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    freqs = np.fft.fftfreq(len(signal), 1/fs)
    spectrum = np.abs(np.fft.fft(signal))
    axs[1].plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])
    axs[1].set_title(f"{title} - Frequency Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")

    st.pyplot(fig)


def plot_constellation(data, title="Constellation Diagram"):
    if data is None:
        st.info("Constellation only available for QAM modulation.")
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(np.real(data[:500]), np.imag(data[:500]), color="blue", s=10, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)


def spectral_efficiency(mod_type, M=16, bandwidth=1e6):
    if mod_type == "QAM":
        bits_per_symbol = np.log2(M)
        efficiency = bits_per_symbol / bandwidth
    elif mod_type == "AM":
        efficiency = 1 / bandwidth
    elif mod_type == "FM":
        efficiency = 0.5 / bandwidth
    else:
        efficiency = 0
    return efficiency


# -------------------------------
# Streamlit App
# -------------------------------

st.title("📡 Radio-over-Fiber (RoF) Link Simulator")
st.markdown("Interactive simulation of RoF optical transmission with modulation, fiber attenuation, noise, and cellular metrics.")

# User controls
mod_type = st.selectbox("Choose Modulation", ["AM", "FM", "QAM"])
fiber_length = st.slider("Fiber Length (km)", 0.1, 50.0, 10.0)
noise_power = st.slider("Noise Power", 0.0, 0.05, 0.01)

M = 16
if mod_type == "QAM":
    M = st.selectbox("Choose QAM Order (M)", [4, 16, 64, 256])

# Generate & transmit
t, tx_signal, data = generate_signal(mod_type, M=M)
rx_signal = fiber_channel(tx_signal, fiber_length, noise_power=noise_power)

# Show plots
plot_time_freq(t, tx_signal, 1e6, title="Transmitted Signal")
plot_time_freq(t, rx_signal, 1e6, title="Received Signal")

# Constellation
if mod_type == "QAM":
    plot_constellation(data, "QAM Constellation (Baseband Symbols)")

# Metrics
snr = 10 * np.log10(np.mean(tx_signal**2) / np.mean((rx_signal - tx_signal)**2))
efficiency = spectral_efficiency(mod_type, M)
st.metric("Estimated SNR (dB)", f"{snr:.2f}")
st.metric("Spectral Efficiency (bits/s/Hz)", f"{efficiency:.4f}")

st.success("✅ Simulation complete. Explore system trade-offs with fiber length, noise, and modulation order.")

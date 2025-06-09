import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal
from scipy.integrate import quad
from math import* 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#TITULOS
st.sidebar.title('Laboratorio 3 - Señales y Sistemas')
st.sidebar.subheader("Nombres: Jessir Florez - Mateo Muñoz - Dylan Abuchaibe")

#SELECCIONADOR DE SERIE Y TRANSFORMADA
serie_checkbox = st.sidebar.checkbox('Mostrar Parte 1: Análisis de Serie de Fourier')
transformada_checkbox = st.sidebar.checkbox('Mostrar Parte 2: Análisis de Transformada de Fourier')

# --- Funciones Auxiliares Globales ---

def calcular_coeficientes_fourier(funcion_a_integrar, num_armonicos_total, periodo_fundamental_integral=2*np.pi):
    """
    Calcula los coeficientes de Fourier An y Bn para una función dada.
    funcion_a_integrar: La función f(x) a analizar (ya debe incluir su amplitud).
    num_armonicos_total: Número total de coeficientes An a calcular (A0, A1, ..., A_{N-1}).
    periodo_fundamental_integral: El periodo sobre el cual se normalizan e integran los coeficientes (default 2*pi).
    Retorna:
        An_coeffs (lista): Coeficientes A0, A1, ..., A_{N-1}.
        Bn_coeffs (lista): Coeficientes B1, B2, ..., B_{N-1}. (longitud N-1)
    """
    An_coeffs = []
    Bn_coeffs_harmonics = [] # Para B1, B2, ... , B_{N-1}
    
    lim_inf = -periodo_fundamental_integral / 2.0
    lim_sup = periodo_fundamental_integral / 2.0
    norm_factor_an = 2.0 / periodo_fundamental_integral
    norm_factor_bn = 2.0 / periodo_fundamental_integral
    
    # A0 (caso especial)
    a0_func = lambda x: funcion_a_integrar(x)
    A0 = quad(a0_func, lim_inf, lim_sup)[0] * norm_factor_an
    An_coeffs.append(A0 / 2.0) # Se almacena A0/2 por convención en la fórmula de reconstrucción

    # An y Bn para n >= 1
    for n_harmonic in range(1, num_armonicos_total): # n_harmonic va de 1 a N-1
        # Factor omega0 * n_harmonic. Si T = 2pi, omega0 = 1. Entonces n_harmonic * x
        # Si T es genérico, omega0 = 2pi/T. Entonces n_harmonic * (2pi/T) * x
        term_angular = n_harmonic * (2 * np.pi / periodo_fundamental_integral)

        func_an = lambda x: funcion_a_integrar(x) * np.cos(term_angular * x)
        an_val = quad(func_an, lim_inf, lim_sup)[0] * norm_factor_an
        An_coeffs.append(an_val)

        func_bn = lambda x: funcion_a_integrar(x) * np.sin(term_angular * x)
        bn_val = quad(func_bn, lim_inf, lim_sup)[0] * norm_factor_bn
        Bn_coeffs_harmonics.append(bn_val)
        
    return An_coeffs, Bn_coeffs_harmonics


def generar_signal(tipo, omega, amplitud, tiempo):
    """Genera diferentes tipos de señales."""
    if tipo == "Senoidal":
        return amplitud * np.sin(omega * tiempo)
    elif tipo == "Cosenoidal":
        return amplitud * np.cos(omega * tiempo)
    elif tipo == "Cuadrada":
        return amplitud * signal.square(omega * tiempo)
    elif tipo == "Diente de Sierra":
        return amplitud * signal.sawtooth(omega * tiempo)
    return np.zeros_like(tiempo)

# --- Fin Funciones Auxiliares Globales ---

# @st.cache_data
def fourier_series_analysis(tipo_signal, num_armonicos, amplitud_entrada, num_periodos_visualizar):
    x_vals = np.arange(-np.pi * num_periodos_visualizar, np.pi * num_periodos_visualizar, 0.001)
    y_signal_original = np.zeros_like(x_vals)
    y_reconstruida = np.zeros_like(x_vals)
    
    phases_harmonics = []

    def funcion_grafica_serie_plotly(x_data, y_original_plot, y_fourier_plot, calculated_phases,
                                     calculated_magnitudes_dc_plus_harmonics, nombre_signal_display, num_harmonics_in_series):
        title_harmonics_count = max(0, num_harmonics_in_series - 1)
        subplot_titles = (
            f"Señal Original vs Aproximación por Serie de Fourier",
            f"Espectro de Fase ({len(calculated_phases)} armónicos)",
            f"Espectro de Magnitud ({title_harmonics_count} armónicos + DC)"
        )
        fig_plotly = make_subplots(rows=3, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.12)
        fig_plotly.add_trace(go.Scatter(x=x_data, y=y_original_plot, mode='lines', name='Señal Original', line=dict(color='blue')), row=1, col=1)
        fig_plotly.add_trace(go.Scatter(x=x_data, y=y_fourier_plot, mode='lines', name='Aproximación Serie Fourier', line=dict(color='red', dash='dash')), row=1, col=1)
        harmonic_numbers_phase = np.arange(1, len(calculated_phases) + 1)
        if len(calculated_phases) > 0 :
            fig_plotly.add_trace(go.Scatter(x=harmonic_numbers_phase, y=calculated_phases, mode='markers', name='Fase de Armónicos', marker=dict(color='green', size=8)), row=2, col=1)
        harmonic_numbers_mag = np.arange(0, len(calculated_magnitudes_dc_plus_harmonics))
        if len(calculated_magnitudes_dc_plus_harmonics) > 0:
            fig_plotly.add_trace(go.Scatter(x=harmonic_numbers_mag, y=calculated_magnitudes_dc_plus_harmonics, mode='markers', name='Magnitud de Armónicos', marker=dict(color='purple', size=8)), row=3, col=1)
        fig_plotly.update_xaxes(title_text="Tiempo (Variable x)", row=1, col=1); fig_plotly.update_yaxes(title_text="Amplitud", row=1, col=1)
        fig_plotly.update_xaxes(title_text="Número de Armónico", type='category', row=2, col=1, tickvals=harmonic_numbers_phase if len(harmonic_numbers_phase)>0 else None)
        fig_plotly.update_yaxes(title_text="Fase (Radianes)", row=2, col=1)
        fig_plotly.update_xaxes(title_text="Número de Armónico (0 para DC)", type='category', row=3, col=1, tickvals=harmonic_numbers_mag if len(harmonic_numbers_mag)>0 else None)
        fig_plotly.update_yaxes(title_text="Magnitud", row=3, col=1)
        main_title_text = (f"Análisis de Serie de Fourier para: Señal {nombre_signal_display}<br>({title_harmonics_count} Armónicos + Componente DC)")
        fig_plotly.update_layout(title_text=main_title_text, title_x=0.5, height=800, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_plotly, use_container_width=True)

    def trapezoidal_signal_func(x_val_in, periodo_base, amp_entrada, duty_cycle_plateau=0.25, slope_duration_factor=0.25):
        # ... (definición sin cambios, omitida por brevedad en este bloque de pensamiento)
        x_norm = np.mod(x_val_in, periodo_base); t_slope = periodo_base * slope_duration_factor; t_plateau = periodo_base * duty_cycle_plateau
        if (2 * t_slope + t_plateau) > periodo_base: t_slope = min(t_slope, periodo_base / 2.0); t_plateau = periodo_base - 2 * t_slope
        t1 = t_slope; t2 = t1 + t_plateau; t3 = t2 + t_slope; y_out = np.zeros_like(x_norm)
        idx_rise = (x_norm > 0) & (x_norm <= t1);
        if t1 > 1e-9: y_out[idx_rise] = (amp_entrada / t1) * x_norm[idx_rise]
        idx_plateau = (x_norm > t1) & (x_norm <= t2); y_out[idx_plateau] = amp_entrada
        idx_fall = (x_norm > t2) & (x_norm <= t3);
        if t_slope > 1e-9: y_out[idx_fall] = amp_entrada - (amp_entrada / t_slope) * (x_norm[idx_fall] - t2)
        return y_out

    periodo_fundamental_calculo = 2 * np.pi
    funcion_a_integrar = None

    if tipo_signal == "Exponencial":
        y_signal_original = amplitud_entrada * np.exp(x_vals)
        funcion_a_integrar = lambda x_lambda: amplitud_entrada * np.exp(x_lambda)
    elif tipo_signal == "Senoidal rectificada":
        y_signal_original = amplitud_entrada * abs(np.sin(x_vals))
        funcion_a_integrar = lambda x_lambda: amplitud_entrada * abs(np.sin(x_lambda))
    elif tipo_signal == "Triangular":
        y_signal_original = amplitud_entrada * signal.sawtooth(x_vals)
        funcion_a_integrar = lambda x_lambda: amplitud_entrada * signal.sawtooth(x_lambda)
    elif tipo_signal == "Rectangular":
        y_signal_original = amplitud_entrada * signal.square(x_vals)
        funcion_a_integrar = lambda x_lambda: amplitud_entrada * signal.square(x_lambda)
    elif tipo_signal == 'Rampa trapezoidal':
        T_base_trapezoid = periodo_fundamental_calculo
        slope_param = 0.2; plateau_param = 0.4
        y_signal_original = trapezoidal_signal_func(x_vals, T_base_trapezoid, amplitud_entrada, duty_cycle_plateau=plateau_param, slope_duration_factor=slope_param)
        funcion_a_integrar = lambda x_int: trapezoidal_signal_func(x_int, T_base_trapezoid, amplitud_entrada, duty_cycle_plateau=plateau_param, slope_duration_factor=slope_param)
    
    if funcion_a_integrar is None: # Fallback
        st.error("Tipo de señal no reconocida para cálculo de coeficientes.")
        return

    # Calcular coeficientes usando la función helper
    # num_armonicos es el N total (A0, A1/B1 ... A_{N-1}/B_{N-1})
    # calcular_coeficientes_fourier devuelve An_coeffs (len N) y Bn_coeffs_harmonics (len N-1)
    An_calc, Bn_calc_harmonics = calcular_coeficientes_fourier(funcion_a_integrar, num_armonicos, periodo_fundamental_calculo)

    # Reconstrucción de la señal
    if num_armonicos > 0 and An_calc:
        y_reconstruida = np.full_like(x_vals, An_calc[0]) # An_calc[0] ya es A0/2
        phases_harmonics = []
        magnitudes_plot = [abs(An_calc[0])] # DC magnitude

        for i in range(1, num_armonicos): # i de 1 a N-1 (para armónicos)
            if i < len(An_calc) and (i-1) < len(Bn_calc_harmonics):
                an_i = An_calc[i]
                bn_i = Bn_calc_harmonics[i-1] # Bn_calc_harmonics es 0-indexed para B1, B2...
                y_reconstruida += (an_i * np.cos(i * x_vals) + bn_i * np.sin(i * x_vals))
                phases_harmonics.append(-np.arctan2(bn_i, an_i))
                magnitudes_plot.append(np.sqrt(an_i**2 + bn_i**2))
            else: # Si no hay suficientes coeficientes (no debería pasar si num_armonicos es correcto)
                phases_harmonics.append(0)
                magnitudes_plot.append(0)
    else:
        y_reconstruida = np.zeros_like(x_vals)
        magnitudes_plot = [0] * num_armonicos if num_armonicos > 0 else []


    if y_signal_original is None: y_signal_original = np.zeros_like(x_vals)

    funcion_grafica_serie_plotly(x_vals, y_signal_original, y_reconstruida,
                                 phases_harmonics, magnitudes_plot, tipo_signal, num_armonicos)

if serie_checkbox:
    st.sidebar.subheader("Configuración Análisis de Serie de Fourier")
    opcion_serie_label = 'Tipo de Señal:'
    opcion_serie = st.sidebar.selectbox(opcion_serie_label, ("Exponencial", "Senoidal rectificada", "Triangular", "Rectangular", "Rampa trapezoidal"), key='select_serie', help="Seleccione el tipo de señal para analizar con Series de Fourier.")
    num_armonicos_label = "Número Total de Armónicos:"
    num_armonicos_input = st.sidebar.number_input(num_armonicos_label, step=1, min_value=1, max_value=100, value=10, key='armonicos_serie', help="Define el número de términos (A0, A1/B1,..., An/Bn) en la serie. Mínimo 1 para DC.")
    amplitud_serie_label = "Amplitud de la Señal Original:"
    amplitud_input_serie = st.sidebar.number_input(amplitud_serie_label, step=1.0, min_value=0.1, max_value=10.0, value=1.0, key='amplitud_serie')
    periodos_label = "Número de Períodos a Visualizar:"
    periodos_input = st.sidebar.number_input(periodos_label, step=1, min_value=1, max_value=10, value=1, key='periodos_serie')
    fourier_series_analysis(opcion_serie, num_armonicos_input, amplitud_input_serie, periodos_input)

# @st.cache_data
def fourier_transform_analysis(num_muestras, frecuencia_muestreo,
                               tipo_s1, freq_s1, amp_s1,
                               tipo_s2, freq_s2, amp_s2,
                               tipo_s3, freq_s3, amp_s3):
    st.title("Análisis de Transformada de Fourier")
    omega1 = 2 * np.pi * freq_s1; omega2 = 2 * np.pi * freq_s2; omega3 = 2 * np.pi * freq_s3
    tiempo_total = 10
    tiempo_vector = np.linspace(0, tiempo_total, num_muestras, endpoint=False)

    # generar_signal ya es global
    y1_signal = generar_signal(tipo_s1, omega1, amp_s1, tiempo_vector)
    y2_signal = generar_signal(tipo_s2, omega2, amp_s2, tiempo_vector)
    y3_signal = generar_signal(tipo_s3, omega3, amp_s3, tiempo_vector)
    signal_compuesta = y1_signal + y2_signal + y3_signal
    fig_transform, axs = plt.subplots(6, 1, figsize=(10, 18))
    plt.subplots_adjust(hspace=1.0)
    axs[0].set_title(f'Señal Componente 1: {tipo_s1} (Frec: {freq_s1} Hz, Amp: {amp_s1})'); axs[0].plot(tiempo_vector, y1_signal, color='blue'); axs[0].set_xlabel('Tiempo (s)'); axs[0].set_ylabel('Amplitud'); axs[0].grid(True)
    axs[1].set_title(f'Señal Componente 2: {tipo_s2} (Frec: {freq_s2} Hz, Amp: {amp_s2})'); axs[1].plot(tiempo_vector, y2_signal, color='green'); axs[1].set_xlabel('Tiempo (s)'); axs[1].set_ylabel('Amplitud'); axs[1].grid(True)
    axs[2].set_title(f'Señal Componente 3: {tipo_s3} (Frec: {freq_s3} Hz, Amp: {amp_s3})'); axs[2].plot(tiempo_vector, y3_signal, color='red'); axs[2].set_xlabel('Tiempo (s)'); axs[2].set_ylabel('Amplitud'); axs[2].grid(True)
    axs[3].set_title('Señal Combinada (Suma de las tres señales componentes)'); axs[3].plot(tiempo_vector, signal_compuesta, color='purple'); axs[3].set_xlabel('Tiempo (s)'); axs[3].grid(True); axs[3].set_ylabel('Amplitud')
    transformada_fourier = np.fft.fft(signal_compuesta) / num_muestras
    transformada_fourier_positiva = transformada_fourier[:num_muestras // 2]
    frecuencias_eje = np.fft.fftfreq(num_muestras, d=1.0/frecuencia_muestreo)[:num_muestras // 2]
    axs[4].set_title('Transformada de Fourier (Espectro de Magnitud)'); axs[4].plot(frecuencias_eje, abs(2 * transformada_fourier_positiva), color='orange'); axs[4].set_xlabel('Frecuencia (Hz)'); axs[4].set_ylabel('Magnitud'); axs[4].grid(True)
    axs[5].set_title('Transformada de Fourier (Espectro de Fase)'); axs[5].phase_spectrum(signal_compuesta, Fs=frecuencia_muestreo, color='magenta'); axs[5].grid(True); axs[5].set_xlabel('Frecuencia (Hz)'); axs[5].set_ylabel('Fase (Radianes)')
    st.pyplot(fig_transform)

if transformada_checkbox:
    st.sidebar.subheader("Configuración General para Transformada de Fourier")
    num_muestras_tf_label = "Número de Muestras para la Señal:"
    num_muestras_input = st.sidebar.number_input(num_muestras_tf_label, step=100, min_value=100, max_value=10000, value=1000, key='num_muestras_tf')
    frecuencia_muestreo_tf_label = "Frecuencia de Muestreo (Hz):"
    frecuencia_muestreo_input = st.sidebar.number_input(frecuencia_muestreo_tf_label, step=50, min_value=100, max_value=5000, value=500, key='freq_muestreo_tf')
    nyquist_freq = frecuencia_muestreo_input / 2.0
    signal_configs = []
    signal_types_espanol = ["Senoidal", "Cosenoidal", "Cuadrada", "Diente de Sierra"]
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuración Señal Componente 1")
    tipo_s1_input = st.sidebar.selectbox("Tipo de Señal 1:", signal_types_espanol, key="tipo_s1_tf")
    freq_s1_input = st.sidebar.number_input("Frecuencia Señal 1 (Hz):", step=1.0, min_value=0.5, max_value=150.0, value=5.0, key='frecuencia_signal1_tf')
    amp_s1_input = st.sidebar.number_input("Amplitud Señal 1:", min_value=0.1, max_value=10.0, step=0.1, value=1.0, key='amplitud_signal1_tf')
    if freq_s1_input >= nyquist_freq: st.warning(f"Advertencia (Señal 1 - {tipo_s1_input}): Frecuencia ({freq_s1_input} Hz) puede causar aliasing. Mitad de frec. muestreo: {nyquist_freq:.2f} Hz.")
    signal_configs.append({'tipo': tipo_s1_input, 'freq': freq_s1_input, 'amp': amp_s1_input})
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuración Señal Componente 2")
    tipo_s2_input = st.sidebar.selectbox("Tipo de Señal 2:", signal_types_espanol, key="tipo_s2_tf", index=1)
    freq_s2_input = st.sidebar.number_input("Frecuencia Señal 2 (Hz):", step=1.0, min_value=0.5, max_value=150.0, value=10.0, key='frecuencia_signal2_tf')
    amp_s2_input = st.sidebar.number_input("Amplitud Señal 2:", min_value=0.1, max_value=10.0, step=0.1, value=1.0, key='amplitud_signal2_tf')
    if freq_s2_input >= nyquist_freq: st.warning(f"Advertencia (Señal 2 - {tipo_s2_input}): Frecuencia ({freq_s2_input} Hz) puede causar aliasing. Mitad de frec. muestreo: {nyquist_freq:.2f} Hz.")
    signal_configs.append({'tipo': tipo_s2_input, 'freq': freq_s2_input, 'amp': amp_s2_input})
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuración Señal Componente 3")
    tipo_s3_input = st.sidebar.selectbox("Tipo de Señal 3:", signal_types_espanol, key="tipo_s3_tf")
    freq_s3_input = st.sidebar.number_input("Frecuencia Señal 3 (Hz):", step=1.0, min_value=0.5, max_value=150.0, value=15.0, key='frecuencia_signal3_tf')
    amp_s3_input = st.sidebar.number_input("Amplitud Señal 3:", min_value=0.1, max_value=10.0, step=0.1, value=1.0, key='amplitud_signal3_tf')
    if freq_s3_input >= nyquist_freq: st.warning(f"Advertencia (Señal 3 - {tipo_s3_input}): Frecuencia ({freq_s3_input} Hz) puede causar aliasing. Mitad de frec. muestreo: {nyquist_freq:.2f} Hz.")
    signal_configs.append({'tipo': tipo_s3_input, 'freq': freq_s3_input, 'amp': amp_s3_input})
    fourier_transform_analysis(num_muestras_input, frecuencia_muestreo_input,
                               signal_configs[0]['tipo'], signal_configs[0]['freq'], signal_configs[0]['amp'],
                               signal_configs[1]['tipo'], signal_configs[1]['freq'], signal_configs[1]['amp'],
                               signal_configs[2]['tipo'], signal_configs[2]['freq'], signal_configs[2]['amp'])

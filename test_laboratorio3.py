import unittest
import numpy as np
from scipy import signal # Needed for signal.square in tests

# Assuming Laboratorio3.py is in the same directory or accessible in PYTHONPATH
import Laboratorio3 as lab3

class TestFourierSeriesCoefficients(unittest.TestCase):

    def test_square_wave_coefficients(self):
        """
        Testa os coeficientes de Fourier para uma onda quadrada simples.
        f(x) = A para 0 < x < pi, -A para -pi < x < 0 (modificada por signal.square que é -A para -pi a 0, A para 0 a pi)
        A0 = 0
        An = 0 para todo n
        Bn = (4*A)/(n*pi) para n ímpar, e 0 para n par.
        """
        A = 1.0
        num_harmonics_test = 5 # A0, A1/B1, A2/B2, A3/B3, A4/B4

        # signal.square(x) é +1 para 0 a pi, -1 para pi a 2pi (ou -pi a 0) com período 2pi
        # Para o intervalo de integração [-pi, pi]:
        #   -1 para x em [-pi, 0)
        #   +1 para x em [0, pi)
        # Isto é o oposto da definição clássica que às vezes se usa (A de 0 a pi).
        # Se f(x) = A para 0 < x < pi and -A para -pi < x < 0: Bn = 4A/(n*pi) for odd n.
        # Se f(x) = -A para -pi < x < 0 and A para 0 < x < pi (como signal.square): Bn = 4A/(n*pi) for odd n.
        # A integral de f(x)*sin(nx) de -pi a pi:
        # int(-A*sin(nx), -pi, 0) + int(A*sin(nx), 0, pi)
        # = -A * [-cos(nx)/n]_{-pi}^0  + A * [-cos(nx)/n]_{0}^{pi}
        # = -A * (-1/n - (-cos(-n*pi)/n)) + A * (-cos(n*pi)/n - (-1/n))
        # = A/n * (1 - cos(n*pi)) + A/n * (1 - cos(n*pi))
        # = 2A/n * (1 - cos(n*pi))
        # Se n é ímpar, cos(n*pi) = -1, então Bn_term = 2A/n * (1 - (-1)) = 4A/n.
        # Se n é par, cos(n*pi) = 1, então Bn_term = 2A/n * (1 - 1) = 0.
        # O fator de normalização no código é (1/pi). Então Bn = (1/pi) * 4A/n = 4A/(n*pi).

        func_square_to_integrate = lambda x: A * signal.square(x) # A amplitude A já está aqui

        # calcular_coeficientes_fourier retorna An (A0/2, A1..), Bn (B1, B2..)
        An_calc, Bn_calc_harmonics = lab3.calcular_coeficientes_fourier(
            func_square_to_integrate,
            num_harmonics_test  # Este é o N total, e.g. 5 significa A0,A1,A2,A3,A4 e B1,B2,B3,B4
        )

        # Test A0 (An_calc[0] é A0/2)
        self.assertAlmostEqual(An_calc[0] * 2.0, 0.0, places=5, msg="A0 deveria ser 0")

        # Test An (A1 a A_{N-1})
        for i in range(1, num_harmonics_test): # i de 1 a 4
            self.assertAlmostEqual(An_calc[i], 0.0, places=5, msg=f"An para n={i} deveria ser 0")

        # Test Bn (B1 a B_{N-1}, que estão em Bn_calc_harmonics[0] a Bn_calc_harmonics[N-2])
        # Bn_calc_harmonics tem N-1 elementos (B1 até B4 se num_harmonics_test=5)
        self.assertEqual(len(Bn_calc_harmonics), num_harmonics_test - 1)

        for n_harmonic_index in range(len(Bn_calc_harmonics)): # n_harmonic_index de 0 a 3
            harmonic_number = n_harmonic_index + 1 # n = 1, 2, 3, 4
            expected_Bn_i = 0.0
            if harmonic_number % 2 != 0: # Ímpar
                expected_Bn_i = (4 * A) / (harmonic_number * np.pi)

            self.assertAlmostEqual(Bn_calc_harmonics[n_harmonic_index], expected_Bn_i, places=5,
                                   msg=f"Bn para n={harmonic_number} não corresponde ao esperado")

class TestSignalGeneration(unittest.TestCase):

    def test_sine_wave_generation(self):
        """Testa a geração de uma onda senoidal."""
        tiempo_test = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        omega_test = 1.0  # Frequência angular de 1 rad/s
        amplitud_test = 2.0

        # Usa o nome em espanhol como definido na lista signal_types_espanol em Laboratorio3.py
        generated_signal = lab3.generar_signal("Senoidal", omega_test, amplitud_test, tiempo_test)
        expected_signal = amplitud_test * np.sin(omega_test * tiempo_test)

        np.testing.assert_array_almost_equal(generated_signal, expected_signal, decimal=5)

    def test_square_wave_generation(self):
        """Testa a geração de uma onda quadrada."""
        tiempo_test = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi - 0.001, 2*np.pi + 0.001]) # Inclui pontos perto da transição
        omega_test = 1.0
        amplitud_test = 1.5

        generated_signal = lab3.generar_signal("Cuadrada", omega_test, amplitud_test, tiempo_test)
        # signal.square(omega*t) tem período 2pi/omega. Para omega=1, período 2pi.
        # Valores esperados para signal.square(t) * amp: A em [0,pi), -A em [pi,2pi)
        expected_signal_scipy = amplitud_test * signal.square(omega_test * tiempo_test)

        np.testing.assert_array_almost_equal(generated_signal, expected_signal_scipy, decimal=5)


if __name__ == '__main__':
    unittest.main()

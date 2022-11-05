import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import signal
from scipy.integrate import quad
from math import* 

#TITULOS
st.sidebar.title('Laboratorio 3 - Señales y sistemas')
st.sidebar.subheader("Jessir Florez - Mateo Muñoz - Dylan Abuchaibe")

#SELECCIONADOR DE SERIE Y TRANSFORMADA
serie = st.sidebar.checkbox('Mostrar parte 1 serie de Fourier.')
transformada = st.sidebar.checkbox('Mostrar parte 2 transformada de Fourier.')

if serie:

    #SELECCIONADOR DE SEÑALES
    opcion = st.sidebar.selectbox(
        'Escoja la señal que desea generar a continuación:',
        ("Exponencial","Senoidal rectificada","Triangular","Rectangular","Rampa trapezoidal"))

    st.sidebar.write('Has seleccionado la función tipo :', opcion)

    armonicos=st.sidebar.number_input("Ingrese el numero de armonicos",step=1,min_value=2,max_value=100,value=2)
    amplitud=st.sidebar.number_input("Ingrese el valor de la amplitud",step=1,min_value=1,max_value=10,value=1)
    periodo=st.sidebar.number_input("Ingrese cuantos periodos desea visualizar",step=1,min_value=1,max_value=10,value=2)
    
    x=np.arange(-np.pi,np.pi,0.001) 

    #DECLARACION DE FUNCIONES
    def funcion_grafica(x,y,sum,fase):
        st.title("Función "+ str(opcion))
        fig,ax=plt.subplots(3)
        fig.set_size_inches(8,7)
        plt.subplots_adjust(hspace=1)
        ax[0].plot(x,sum,"r--")
        ax[0].plot(x,y,"b")
        ax[1].phase_spectrum(sum[:50])
        ax[2].magnitude_spectrum(sum[:50])
        ax[1].grid(True)
        ax[2].grid(True)
        ax[1].set_title("Gráfico de fase con " + str(armonicos) + " armonicos")
        ax[2].set_title("Gráfico de magnitud con " + str(armonicos) + " armonicos")
        ax[0].set_title("Serie de fourier con " + str(armonicos) + " armonicos")
        ax[0].legend(['Fourier', 'Original'])
        ax[0].set_ylabel("Eje y")
        ax[0].grid(True)
        st.pyplot(fig)

    # FUNCION EXPONENCIAL
    if opcion == "Exponencial":

        y= amplitud*np.exp(x) 

        An=[] 
        Bn=[]

        fun_cos=lambda x: np.exp(x)*cos(i*x)  
        fun_sen=lambda x: np.exp(x)*sin(i*x)

        sum=0

        for i in range(armonicos):
            an=quad(fun_cos,-np.pi,np.pi)[0]*(1.0/np.pi)
            An.append(an*amplitud)

        for i in range(armonicos):
            bn=quad(fun_sen,-np.pi,np.pi)[0]*(1.0/np.pi)
            Bn.append(bn*amplitud) 

        for i in range(armonicos):
            if i==0.0:
                sum=sum+An[i]/2
            else:
                sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))
            
            phase=(-np.arctan(sum)**-1)
    
            
        funcion_grafica(x,y,sum,phase)

    # FUNCION SENOIDAL RECTIFICADA
    elif opcion == "Senoidal rectificada":
            
        y=abs(np.sin(np.pi*x))
        y=y*amplitud

        An=[] 
        Bn=[]

        fun_cos=lambda x: amplitud*abs(np.sin(np.pi*x))*cos(i*x)  
        fun_sen=lambda x: amplitud*abs(np.sin(np.pi*x))*sin(i*x)

        sum=0

        for i in range(armonicos):
            an=quad(fun_cos,-np.pi,np.pi)[0]*(1.0/np.pi)
            An.append(an)

        for i in range(armonicos):

            bn=quad(fun_sen,-np.pi,np.pi)[0]*(1.0/np.pi)
            Bn.append(bn) 

        for i in range(armonicos):
            if i==0.0:
                sum=sum+An[i]/2
                
            else:
                sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))
                phase=(np.arctan(sum)*-1)

                
        funcion_grafica(x,y,sum,phase)

    #FUNCION TRIANGULAR
    elif opcion == "Triangular":
    
        y = amplitud*signal.sawtooth(2*np.pi*x)

        An=[] 
        Bn=[]

        fun_cos=lambda x: amplitud*signal.sawtooth(2*np.pi*x)*cos(i*x)  
        fun_sen=lambda x: amplitud*signal.sawtooth(2*np.pi*x)*sin(i*x)

        sum=0

        for i in range(armonicos):
            an=quad(fun_cos,-np.pi,np.pi)[0]*(1.0/np.pi)
            An.append(an)

        for i in range(armonicos):

            bn=quad(fun_sen,-np.pi,np.pi)[0]*(1.0/np.pi)
            Bn.append(bn) 

        for i in range(armonicos):
            if i==0.0:
                sum=sum+An[i]/2
                
            else:
                sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))
                phase=(np.arctan(sum)**-1)

                
        funcion_grafica(x,y,sum,phase)

    # FUNCION CUADRADA
    elif opcion == "Rectangular":

        y = amplitud*signal.square(2*np.pi*x)

        An=[] 
        Bn=[]

        fun_cos=lambda x: amplitud*signal.square(2*np.pi*x)*cos(i*x)  
        fun_sen=lambda x: amplitud*signal.square(2*np.pi*x)*sin(i*x)

        sum=0

        for i in range(armonicos):
            an=quad(fun_cos,-np.pi,np.pi)[0]*(1.0/np.pi)
            An.append(an)

        for i in range(armonicos):

            bn=quad(fun_sen,-np.pi,np.pi)[0]*(1.0/np.pi)
            Bn.append(bn) 

        for i in range(armonicos):
            if i==0.0:
                sum=sum+An[i]/2
                
            else:
                sum=sum+(An[i]*np.cos(i*x)+Bn[i]*np.sin(i*x))
                phase=(-np.arctan(sum)*-1)

        funcion_grafica(x,y,sum,phase)

    # RAMPA TRAPEZOIDAL
    if opcion == 'Rampa trapezoidal':
        
        fig,ax=plt.subplots()

        def trapzoid_signal(x, width=2., slope=1., amp=1., offs=0):
            a = slope*width*signal.sawtooth(2*np.pi*x/width, width=0.5)/4.
            a += slope*width/4.
            a[a>amp] = amp
            return a + offs

        for w,s,a in zip([10], [1], [3.25]):
            x = np.linspace(0, w, 501)
            l = "width={}, slope={}, amp={}".format(w,s,a)

        y=trapzoid_signal(x, width=w, slope=s, amp=a)
        ax.plot(x,y)
        ax.set_title("Serie de fourier con " + str(armonicos) + " armonicos")
        ax.grid(True)

        st.pyplot(fig)


#ETAPA TRANSFORMADA DE FOURIER

if transformada:

    st.title("Transformada de Fourier")

    num= st.sidebar.number_input("Ingrese el numero de muestras que desea tomar: ",
    step=10,min_value=1,max_value=10000,value=50)

    samplingFrequency= st.sidebar.number_input("Ingrese el valor de la frecuencia de muestreo: ",
    step=10,min_value=1,max_value=1000,value=100)

    samplingInterval= 1/samplingFrequency


# INGRESE EL VALOR DE LAS FRECUENCIAS DE CADA SEÑAL
    signal1Frequency=st.sidebar.number_input("Ingrese el valor de frecuencia de la señal 1 ",
    step=1,min_value=1,max_value=150,value=3)

    signal2Frequency=st.sidebar.number_input("Ingrese el valor de frecuencia de la señal 2 ",
    step=1,min_value=1,max_value=130,value=5)

    signal3Frequency=st.sidebar.number_input("Ingrese el valor de frecuencia de la señal 3 ",
    step=1,min_value=1,max_value=140,value=7)

    wo1=2*np.pi*signal1Frequency
    wo2=2*np.pi*signal2Frequency
    wo3=2*np.pi*signal3Frequency

    time= np.linspace(0,10,num)
    
    y1 = np.sin(wo1*time)
    y2 = np.cos(wo2*time)
    y3 = np.sin(wo3*time)

    a1= st.number_input("Ingrese el valor de amplitud para la señal número 1: ",
    min_value=0,max_value=10,step=1,value=1)

    a2= st.number_input("Ingrese el valor de amplitud para la señal número 2: ",
    min_value=0,max_value=10,step=1,value=2)

    a3= st.number_input("Ingrese el valor de amplitud para la señal número 3: ",
    min_value=0,max_value=10,step=1,value=3)

    y1=y1*a1
    y2=y2*a2
    y3=y3*a3

    figure, axis = plt.subplots(3)
    plt.subplots_adjust(hspace=.5)
    figure.set_size_inches(8,7)

    amplitude = y1+y2+y3

    axis[0].set_title('Sine wave with multiple frequencies')
    axis[0].plot(time, amplitude)
    axis[0].set_xlabel('Time')
    axis[0].grid(True)
    axis[0].set_ylabel('Amplitude')
    
    fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(amplitude)))] # Exclude sampling frequency

    tpCount= len(amplitude)
    values = np.arange(int(tpCount))
    timePeriod  = tpCount/samplingFrequency
    frequencies = values/timePeriod

    axis[1].set_title('Fourier transform depicting the frequency components')
    axis[1].plot(frequencies, abs(2*fourierTransform))
    axis[1].set_xlabel('Frequency')
    axis[1].set_ylabel('Amplitude')
    axis[1].grid(True)
    axis[2].grid(True)
    axis[2].phase_spectrum(amplitude)

    st.pyplot(figure)

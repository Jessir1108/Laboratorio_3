import numpy as np
import matplotlib.pyplot as fig
import streamlit as st
from scipy import signal
from scipy.integrate import quad
from math import* 

#TITULOS
st.sidebar.title('Laboratorio 3 - Señales y sistemas')
st.sidebar.subheader("Jessir Florez - Mateo Muñoz - Dylan Abuchaibe")

#SELECCIONADOR DE SEÑALES
opcion = st.sidebar.selectbox(
     'Escoja la señal que desea generar a continuación:',
     ("Exponencial","Senoidal rectificada","Triangular","Rectangular","Rampa trapezoidal"))

st.sidebar.write('Has seleccionado la función tipo :', opcion)

armonicos=st.sidebar.number_input("Ingrese el numero de armonicos",step=1,min_value=2,max_value=100,value=2)
amplitud=st.sidebar.number_input("Ingrese el valor de la amplitud",step=1,min_value=1,max_value=10,value=1)
x=np.arange(-np.pi,np.pi,0.001) 

#DECLARACION DE FUNCIONES
def funcion_grafica(x,y,sum):
    st.title("Función "+ str(opcion))
    fig,ax=fig.subplots()
    ax.plot(x,sum,"r--")
    fig.plot(x,y,"b")
    ax.set_title("Serie de fourier con " + str(armonicos) + " armonicos")
    ax.legend(['Fourier', 'Original'])
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.grid(True)
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
            
    funcion_grafica(x,y,sum)

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
            
    funcion_grafica(x,y,sum)

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
            
    funcion_grafica(x,y,sum)

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

    funcion_grafica(x,y,sum)

if opcion == 'Rampa trapezoidal':
    s=0.001
    xinicio = 2
    xfinal = 10
    if xfinal>xinicio:
        e=(xfinal-xinicio)/3
        def tramo1(z):         
            return z-xinicio    
        def tramo2(z):         
            return e    
        def tramo3(z):         
            return -z+xfinal     
        a=xinicio
        b=xinicio+e   
        c=xinicio+2*e
        d=xinicio+3*e
        t= np.arange(xinicio, xfinal+s, s)
        y=np.piecewise(t,[(a<=t) & (t<b),(b<=t)&(t<=c),(c<t)&(t<=d)],[lambda t:tramo1(t),lambda t: tramo2(t),lambda t:tramo3(t)])    
        tramo1=np.vectorize(tramo1)     
        fig.plot(t[t<b],tramo1(t[t<b]),c="c")  
        tramo2=np.vectorize(tramo2) 
        fig.plot(t[(b<=t)&(t<c)],tramo2(t[(b<=t)&(t<c)]),c="c") 
        tramo3=np.vectorize(tramo3)     
        fig.plot(t[(c<=t)&(t<=d)],tramo3(t[(c<=t)&(t<=d)]),c="c")  
        fig.xlabel("t(s)")
        fig.ylabel("x(t)")
        fig.title("Gráfica x(t)")
        st.pyplot(fig)

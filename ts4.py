#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/9/2021

@author: Gisela Farace

Descripción: Tarea semanal 4
------------
"""

# Importación de módulos para Jupyter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Funciones

def senoidal(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs # tiempo de muestreo
    df = fs/nn # resolución espectral 
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    
    # grilla de sampleo frecuencial
    sen = vmax*np.sin(2*np.pi*ff*tt + ph)+dc
    
    return tt, sen

def cuantizacion(B, Vref, xx):
    q = Vref/2**(B) #Volts
    x = np.round(xx/q)
    xq = x*q
    error = xq - xx
    
    return xq, error, q

def fft(xx, N):
    ft = np.fft.fft(xx, axis=0)
    ft = ft/N
    return ft

def analisis_adc(xx, tt, fs, N, B, Vref, kn):
    df = fs/N # resolución espectral
         
    #ADC
    q = Vref/2**(B) #Volts
    
    # Datos ruido
    pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)
    desv = np.sqrt(pot_ruido)
    # Ruido incorrelado y gaussiano
    n = np.random.normal(0,desv, size=N)
    # Señal contaminada con ruido
    sr = analog_sig + n
    #ADC
    srq, nq, q = cuantizacion(B, Vref, sr)
    
    # Presentación gráfica de los resultados  
    fig, axs = plt.subplots(3)
    axs[0].plot(tt, srq, lw=2)
    axs[0].plot(tt, sr, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none')
    axs[0].plot(tt, analog_sig, color='orange', ls='dotted')
    axs[0].legend(['$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)','$ s_R = s + n $  (ADC in)', '$ s $ (analog)'])
    axs[0].set_title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - kn = {:3.1f}'.format(B, Vref, q, kn) )
    axs[0].set(xlabel='tiempo [segundos]', ylabel='Amplitud [V]')
    axes_hdl = plt.gca()
     
    # Densidad espectral de potencia
    ff = np.linspace(0, (N-1), N)*df
    ff_os = ff
    ft_Srq = fft(srq,N)
    ft_As = fft(analog_sig,N)
    ft_SR = fft(sr,N)
    ft_Nn = fft(n,N)
    ft_Nq = fft(nq,N)
    
    nNn_mean = np.mean(np.abs(ft_Nn)**2)
    Nnq_mean = np.mean(np.abs(ft_Nq)**2)
    
    bfrec = ff <= fs/2
    
    axs[1].plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2 )
    axs[1].plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_As[ff_os <= fs/2])**2), color='orange', ls='dotted' )
    axs[1].plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g')
    axs[1].plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_Nn[ff_os <= fs/2])**2), ':r')
    axs[1].plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
    axs[1].plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r')
    axs[1].plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c')
    axs[1].set_title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V- kn = {:3.1f}'.format(B, Vref, q, kn))
    axs[1].legend(['$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)','$ s $ (analog)','$ s_R = s + n $  (ADC in)','$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)),'$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) ])
    axs[1].set(xlabel='Frecuencia [Hz]', ylabel='Densidad de Potencia [dB]')
    axes_hdl = plt.gca()
    # suponiendo valores negativos de potencia ruido en dB
    axs[1].set_ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))
    
    # Histograma
    bins = 10
    axs[2].hist(nq, bins=bins)
    axs[2].plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
    axs[2].set_title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - kn = {:3.1f}'.format(B, Vref, q, kn))
    
    return

plt.close('all')

# datos de la senoidal
vmax = 1
dc = 0
ff = 1
ph = 0
N = 1000  # cantidad de muestras
fs = 1000 # frecuencia de muestreo (Hz)

# Datos del ADC
Vref = 2 # Volts
B = 4

# datos del ruido
kn = 1

# Senoidal
tt,analog_sig = senoidal(vmax, dc, ff, ph, N, fs)

analisis_adc(analog_sig, tt, fs, N, B, Vref, kn)

B=4
kn=1/10
analisis_adc(analog_sig, tt, fs, N, B, Vref, kn)

B=4
kn=10
analisis_adc(analog_sig, tt, fs, N, B, Vref, kn)

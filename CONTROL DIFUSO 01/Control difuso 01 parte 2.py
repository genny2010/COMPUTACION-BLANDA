#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Control Difuso
# Maria Camila Muñoz Mejia
#Ingenieria de sistemas y computación
#Computación blanda


import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_calidad = np.arange (0,11,1)
x_servicio = np.arange (0,11,1)
x_propina = np.arange (0,26,1)

calidad_baja = fuzz.trimf(x_calidad,[0,0,5])
calidad_media = fuzz.trimf(x_calidad,[0,5,10])
calidad_alta = fuzz.trimf(x_calidad,[5,10,10])
servicio_baja = fuzz.trimf(x_servicio,[0,0,5])
servicio_media = fuzz.trimf(x_servicio,[0,5,10])
servicio_alta = fuzz.trimf(x_servicio,[5,10,10])
propina_baja = fuzz.trimf(x_propina,[0,0,13])
propina_media = fuzz.trimf(x_propina,[0,13,25])
propina_alta = fuzz.trimf(x_propina,[13,25,25])

fig, (ax0,ax1,ax2) = plt.subplots(nrows=3,figsize =(8,9))
ax0.plot(x_calidad, calidad_baja, 'b', linewidth=1.5, label= 'mala')
ax0.plot(x_calidad,calidad_media, 'g', linewidth=1.5, label= 'aceptable')
ax0.plot(x_calidad,calidad_alta, 'r', linewidth=1.5, label= 'buena')
ax0.set_title('calidad del la comida')
ax0.legend()

ax1.plot(x_servicio,servicio_baja, 'b', linewidth=1.5, label= 'mala')
ax1.plot(x_servicio,servicio_media, 'g', linewidth=1.5, label= 'aceptable')
ax1.plot(x_servicio,servicio_alta, 'r', linewidth=1.5, label= 'buena')
ax1.set_title('calidad del servicio')
ax1.legend()

ax2.plot(x_propina,propina_baja, 'b',linewidth=1.5, label= 'mala')
ax2.plot(x_propina,propina_media, 'g', linewidth=1.5, label= 'aceptable')
ax2.plot(x_propina,propina_alta, 'r', linewidth=1.5, label= 'buena')
ax2.set_title('valor de la propina')
ax2.legend()
        
 
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)         
    ax.get_xaxis().tick_bottom()                  
    ax.get_yaxis().tick_left()
plt.tight_layout()


# In[ ]:





# In[ ]:





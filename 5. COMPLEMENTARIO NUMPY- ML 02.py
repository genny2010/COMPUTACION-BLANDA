#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Genny Paola Rivera Becerra / codigo: 1087561571 / Computación Blanda / Ingenieria de sistemas y computación 

import numpy as np

a = np.arange(9).reshape(3,3)

print('a =\n', a, '\n')

b = a*3

print('b =\n', b)


# In[2]:


print('a =\n', a, '\n')

print('b =\n', b, '\n')

print('Apilamiento horizontal =\n', np.hstack((a,b)) )


# In[3]:


print('a =\n', a, '\n')

print('b =\n', b, '\n')

print( 'Apilamiento horizontal con concatenate = \n',
            np.concatenate((a,b), axis=1) )


# In[4]:


print('a =\n', a, '\n')

print('b =\n', b, '\n')

print( 'Apilamiento vertical =\n', np.vstack((a,b)) )


# In[5]:


print('a =\n', a, '\n')

print('b =\n', b, '\n')

print( 'Apilamiento vertical con concatenate =\n',
            np.concatenate((a,b), axis=1) )


# In[6]:


print('a =\n', a, '\n')

print('b =\n', b, '\n')

print( 'Apilamiento en profundidad =\n', np.dstack((a,b)) )


# In[7]:


print('a =\n', a, '\n')
 
print('b =\n', b, '\n')

print( 'Apilamiento por columnas =\n',
            np.column_stack((a,b)) )


# In[8]:


print('a =\n', a, '\n')

print('b =\n', b, '\n')

print( 'Apilamiento por filas =\n',
            np.row_stack((a,b)) )


# In[9]:


print(a, '\n')

print('Array con división horizontal =\n', np.hsplit(a, 3), '\n')

print('Array con división horizontal, uso de split() =\n',
        np.split(a, 3, axis=0))


# In[10]:


print(a, '\n')

print('División Vertical = \n', np.vsplit(a, 3), '\n')

print('Array con división vertical, uso de split() =\n',
            np.split(a, 3, axis=1))


# In[11]:


c = np.arange(27).reshape(3, 3, 3)

print(c, '\n')

print('División en profundidad =\n', np.dsplit(c,3), '\n')


# In[12]:


print(b, '\n')

print('ndim: ', b.ndim)


# In[13]:


print(b, '\n')

print('size: ', b.size)


# In[14]:


print('itemsize: ', b.itemsize)


# In[15]:


print(b, '\n')

print('nbytes: ', b.nbytes, '\n')

print('nbytes equivalente: ', b.size * b.itemsize)


# In[30]:


b = np.array([1.j + 1, 2.j + 3])

print('Complejo: \n', b)


# In[31]:


b.resize(6,4)

print(b, '\n')

print('Transpuesta: ', b.T)


# In[32]:


print('real: ', b.real, '\n')

print('imaginario: ', b.imag)


# In[19]:


print(b.dtype)


# In[24]:


b = np.arange(4).reshape(2,2)

print(b, '\n')

f = b.flat

print(f, '\n')

for item in f: print (item)

print('\n')

print('Elemento 2: ', b.flat[2])

b.flat = 7

print(b, '\n')

b.flat[[1,3]] = 1

print(b, '\n')


# In[ ]:





#TALLER MATPLOTLIB

import numpy as np
import matplotlib.pyplot as plt

#1. GENERAR GRAFICAS 
#A)
x = np.linspace(0, 2*np.pi, 100)

y_seno = np.sin(x)
y_coseno = np.cos(x)

plt.figure() 

plt.plot(x, y_seno, 'm--')  

plt.title('Función seno')
plt.xlabel('Ángulo(rad)')      
plt.ylabel('Sin(x)')

plt.show() 

#B)

plt.figure() 

plt.plot(x, y_seno, 'm--', label='Función seno')

plt.plot(x, y_coseno, 'g-', label='Función coseno')

plt.xlabel('Ángulo(rad)')
plt.ylabel('Valor')
plt.legend()

plt.show()

# %%
#2. HISTOGRAMA
datos = np.random.randn(100000)

plt.figure()


plt.hist(datos, bins=30, density=True, color='pink', edgecolor='purple', alpha=0.4)

x = np.linspace(datos.min(), datos.max(), 100)
y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)  

plt.plot(x, y, 'm--', linewidth=1.5) 


plt.title('Histograma')
plt.xlabel('Valor x')
plt.ylabel('Probabilidad')

plt.show()
# %%

#3) GRAFICAR 


x = np.linspace(-5, 5, 200)
#a)
y1 = 2*x + 1

#b)
y2 = -x**2

plt.figure()

plt.plot(x, y1, 'b-', label='f(x) = 2x + 1')   
plt.plot(x, y2, 'r-', label='g(x) = -x²')   

plt.title('Gráficas de 2x+1 y -x²')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)  

plt.show()

# %%
#EJERCICIO GITHUB
#1)
vector = np.random.rand(720)
print("primer paso", vector[:10])

#2)
matriz = vector.reshape(120, 6)
print("segundo paso", matriz[:5])

#3)
matriz_T = matriz.T
print("tercer paso", matriz[5:])
# Copia en orden C (fila por fila)
matriz_C = np.array(matriz_T, order='C', copy=True)

# Copia en orden F (columna por columna)
matriz_F = np.array(matriz_T, order='F', copy=True)
print("Copia en orden C:")
print(matriz_C.flags)

print("\nCopia en orden F:")
print(matriz_F.flags)


fig = plt.figure(figsize=(15, 10))

# Panel 1 - Plot (línea)
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(matriz_T[0], color='blue', label='Línea')
ax1.set_title('1) Plot - Línea')
ax1.set_xlabel('Índice')
ax1.set_ylabel('Valor')
ax1.legend()
ax1.grid(True)

#  Panel 2 - Scatter (dispersión)
ax2 = fig.add_subplot(3, 2, 2)
ax2.scatter(np.arange(len(matriz_T[1])), matriz_T[1], color='red', s=15, label='Puntos')
ax2.set_title('2) Scatter')
ax2.set_xlabel('Índice')
ax2.set_ylabel('Valor')
ax2.legend()
ax2.grid(True)

#Panel 3 - Bar (barras)
ax3 = fig.add_subplot(3, 2, 3)
indices = np.arange(10)  # Usamos solo 10 barras para que se vea bien
valores_bar = matriz_T[2][:10]
ax3.bar(indices, valores_bar, color='green', label='Barras')
ax3.set_title('3) Bar')
ax3.set_xlabel('Categoría')
ax3.set_ylabel('Valor')
ax3.legend()
ax3.grid(True)

#Panel 4 - Histograma
ax4 = fig.add_subplot(3, 2, 4)
ax4.hist(matriz_T[3], bins=15, color='orange', edgecolor='black', alpha=0.7, label='Hist')
ax4.set_title('4) Histograma')
ax4.set_xlabel('Valor')
ax4.set_ylabel('Frecuencia')
ax4.legend()
ax4.grid(True)

# Panel 5 - Pie (pastel)
ax5 = fig.add_subplot(3, 2, 5)
valores_pie = matriz_T[4][:6]
valores_pie = valores_pie / valores_pie.sum() 
ax5.pie(valores_pie, labels=[f'P{i+1}' for i in range(6)], autopct='%1.1f%%')
ax5.set_title('5) Pie chart')

#Panel 6 - Boxplot
ax6 = fig.add_subplot(3, 2, 6)
ax6.boxplot(matriz_T[5], vert=True, patch_artist=True)
ax6.set_title('6) Boxplot')
ax6.set_ylabel('Valor')
ax6.grid(True)

plt.tight_layout()
plt.show()




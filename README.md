# Automatización de seguimiento de rutina de gimnasio

##  Planteamiento del problema
En mi día a día, llevar el control de las repeticiones de cada ejercicio en el gimnasio es una tarea repetitiva y tediosa.  
Revisar manualmente semana por semana, calcular cuántas repeticiones faltan para alcanzar la meta y generar gráficas comparativas consume tiempo y es propenso a errores.

##  Objetivo
Automatizar el análisis de mi progreso en el gimnasio:
- Leer automáticamente los datos de repeticiones desde un archivo CSV.
- Calcular cuántas repeticiones faltan para llegar a la meta.
- Clasificar el estado de cada ejercicio (Completado / Pendiente).
- Evaluar el nivel de progreso (Lejos, Cerca, Muy cerca, Meta alcanzada).
- Generar gráficas comparativas entre semanas y contra la meta final.
- Guardar los resultados en archivos listos para consultar.

##  Algoritmo de la solución
1. **Lectura de datos**: se carga un archivo CSV con los ejercicios, repeticiones sugeridas y repeticiones realizadas por semana.
2. **Conversión de datos**: se convierten las columnas de semanas a valores numéricos para poder operar con ellas.
3. **Cálculo de faltantes**: se resta el valor de cada semana contra las repeticiones sugeridas.
4. **Clasificación del estado**:
   - `Completado` si la última semana alcanzó la meta.
   - `Pendiente` si aún no se alcanza.
5. **Clasificación del progreso**:
   - ≤ 13 → Lejos de la meta  
   - 14–17 → Cerca de la meta  
   - 18–19 → Muy cerca de la meta  
   - 20 → Meta alcanzada
6. **Comparaciones dinámicas**: se pueden comparar dos semanas específicas o mostrar solo una semana seleccionada.
7. **Visualización**: se generan gráficas de barras:
   - Comparación entre dos semanas.
   - Comparación final entre la última semana y las repeticiones sugeridas.
8. **Exportación**: las gráficas se guardan automáticamente en la carpeta `out/`.

##  Ejemplo de salida
- Tabla con columnas: Día, Ejercicio, Repeticiones sugeridas, Semana X, Faltan, Estado, Progreso.
- Gráficas guardadas en `out/` mostrando evolución y comparación final.

##  Cómo ejecutar
1. Guardar el archivo `rutina.py` en tu proyecto.
2. Colocar el archivo `rutina.0.csv` en la carpeta `inputs/`.
3. Ejecutar en consola:
   ```bash
   python rutina.py
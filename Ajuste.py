import streamlit as st
import pandas as pd
import numpy as np
from scipy import odr
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Ajuste ODR con Errores XY", layout="wide")

def main():
    st.title("Ajuste de Curvas con Errores en X e Y (ODR)")
    st.markdown("""
    Esta aplicación utiliza **Regresión de Distancia Ortogonal (ODR)** para ajustar modelos 
    teniendo en cuenta la incertidumbre tanto en la variable dependiente ($y$) como en la independiente ($x$).
    """)

    # --- 1. DATOS DE ENTRADA ---
    st.sidebar.header("1. Configuración de Datos")
    
    # Datos de ejemplo iniciales
    default_data = pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [2.1, 3.9, 6.2, 8.1, 9.8],
        'x_err': [0.1, 0.1, 0.2, 0.1, 0.2],
        'y_err': [0.2, 0.3, 0.2, 0.4, 0.3]
    })

    input_method = st.sidebar.radio("Método de entrada:", ["Editar Tabla", "Subir CSV"])

    if input_method == "Subir CSV":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = default_data
            st.sidebar.info("Usando datos de ejemplo.")
    else:
        st.sidebar.info("Edita los valores directamente en la tabla.")
        df = default_data

    # Editor de datos editable
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Datos")
        edited_df = st.data_editor(df, num_rows="dynamic")
    
    # Validación básica
    required_cols = {'x', 'y', 'x_err', 'y_err'}
    if not required_cols.issubset(edited_df.columns):
        st.error(f"El dataset debe contener las columnas: {required_cols}")
        return

    # Convertir a arrays de numpy para procesamiento
    x = edited_df['x'].to_numpy(dtype=float)
    y = edited_df['y'].to_numpy(dtype=float)
    sx = edited_df['x_err'].to_numpy(dtype=float)
    sy = edited_df['y_err'].to_numpy(dtype=float)

    # Evitar errores 0 para no causar divisiones por cero en los pesos
    # Evitar errores 0 para no causar divisiones por cero en los pesos
    # En lugar de fijar un valor absoluto, usar un valor proporcional
    # al propio punto (mucho menor que el valor) para respetar el
    # orden de magnitud de los datos. Permitimos que el usuario ajuste
    # esta fracción desde la barra lateral.
    rel_fraction = st.sidebar.number_input(
        "Fracción relativa para errores cero",
        min_value=1e-12,
        max_value=1e-2,
        value=1e-6,
        format="%.12e",
        help="Fracción de |valor| usada cuando el error es 0. Por ejemplo 1e-6"
    )

    sx_zero_mask = sx == 0
    if np.any(sx_zero_mask):
        # Si x != 0: usar una fracción relativa |x| * rel_fraction
        # Si x == 0: usar un mínimo absoluto muy pequeño
        sx[sx_zero_mask] = np.where(
            np.abs(x[sx_zero_mask]) > 0,
            np.abs(x[sx_zero_mask]) * rel_fraction,
            1e-12
        )

    sy_zero_mask = sy == 0
    if np.any(sy_zero_mask):
        sy[sy_zero_mask] = np.where(
            np.abs(y[sy_zero_mask]) > 0,
            np.abs(y[sy_zero_mask]) * rel_fraction,
            1e-12
        )

    # --- 2. CONFIGURACIÓN DEL MODELO ---
    with col2:
        st.subheader("Ajuste del Modelo")
        degree = st.selectbox(
            "Selecciona el tipo de ajuste:",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Lineal (Grado 1)", 2: "Cuadrático (Grado 2)", 3: "Cúbico (Grado 3)", 4: "Grado 4"}[x]
        )

        if st.button("Calcular Ajuste"):
            run_odr_fit(x, y, sx, sy, degree)

def run_odr_fit(x, y, sx, sy, degree):
    # Definición del modelo polinómico dinámico
    def polynomial_func(B, x):
        # B es un array de coeficientes. 
        # Si grado 1 (lineal): B[0]*x + B[1]
        # np.polyval espera los coeficientes de mayor a menor grado
        return np.polyval(B, x)

    # Inicializar modelo ODR
    # poly_model = odr.Model(polynomial_func) # Usamos modelo custom
    # Pero scipy.odr tiene un modelo polinómico predefinido que es más rápido:
    poly_model = odr.polynomial(degree)

    # Crear objeto Data. ODR usa pesos (weights) que son 1/error^2
    my_data = odr.RealData(x, y, sx=sx, sy=sy)

    # Estimación inicial (beta0). OLS simple para tener un punto de partida
    # polyfit devuelve coeficientes de mayor a menor grado, perfecto para odr.polynomial
    initial_guess = np.polyfit(x, y, degree)

    # Ejecutar ODR
    my_odr = odr.ODR(my_data, poly_model, beta0=initial_guess)
    output = my_odr.run()

    # --- 3. RESULTADOS ---
    display_results(output, degree)
    plot_results(x, y, sx, sy, output, degree)

def display_results(output, degree):
    st.markdown("---")
    st.subheader("Resultados del Ajuste")

    # Métricas de bondad de ajuste
    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-Cuadrado", f"{output.sum_square:.4f}")
    c2.metric("Grados de Libertad", f"{output.iwork[10]}") # iwork[10] suele contener DoF en ODRPACK
    c3.metric("Razón de varianza (Chi_sq_red)", f"{output.res_var:.4f}")

    st.markdown("#### Parámetros ($B$) y sus Errores Estándar ($\sigma$)")
    
    # Crear tabla de resultados bonita
    params = output.beta
    errors = output.sd_beta
    
    # Nombres de coeficientes según el grado
    # Nota: odr.polynomial usa orden ascendente o descendente según versión, 
    # pero np.polyval y polyfit usan: c_n * x^n + ... + c_0
    # Scipy ODR polynomial model: B[0] es el término de mayor grado.
    
    param_names = []
    for i in range(degree, -1, -1):
        if i == 0: param_names.append("Intersección (Term Indep.)")
        elif i == 1: param_names.append("Pendiente (x)")
        else: param_names.append(f"Coeficiente x^{i}")

    results_df = pd.DataFrame({
        "Parámetro": param_names,
        "Valor Estimado": params,
        "Error Estándar (+/-)": errors,
        "Error Relativo (%)": np.abs(errors/params) * 100
    })
    
    st.table(results_df.style.format({
        "Valor Estimado": "{:.6f}", 
        "Error Estándar (+/-)": "{:.6f}",
        "Error Relativo (%)": "{:.2f}%"
    }))

    st.info("Nota: El algoritmo tiene en cuenta los errores en X e Y para calcular la incertidumbre de estos parámetros (Matriz de covarianza).")

def plot_results(x, y, sx, sy, output, degree):
    # Generar línea de ajuste suave
    x_range = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_fit = np.polyval(output.beta, x_range)

    # Crear gráfico interactivo con Plotly
    fig = go.Figure()

    # 1. Puntos de datos con barras de error
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Datos Experimentales',
        error_x=dict(type='data', array=sx, visible=True),
        error_y=dict(type='data', array=sy, visible=True),
        marker=dict(color='red', size=8)
    ))

    # 2. Curva de ajuste
    fig.add_trace(go.Scatter(
        x=x_range, y=y_fit,
        mode='lines',
        name=f'Ajuste ODR (Grado {degree})',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title="Visualización del Ajuste (Zoom interactivo)",
        xaxis_title="Eje X",
        yaxis_title="Eje Y",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
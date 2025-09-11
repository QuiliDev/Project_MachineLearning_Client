import streamlit as st

st.set_page_config(page_title="Ayuda / About", page_icon="❓", layout="centered")

st.title("❓ Ayuda / About")
st.caption("Guía rápida para entender el modelo, los umbrales y los campos del formulario.")

st.header("¿Qué predice el modelo?")
st.markdown(
"""
Este modelo estima la **probabilidad de que una queja termine en disputa**.
Se entrena con históricos de quejas y devuelve un valor entre **0 y 1** (0–100%).
"""
)

st.header("Umbrales de decisión")
st.markdown(
"""
- **Equilibrado (F1):** balance entre *precision* y *recall*. Recomendado por defecto.  
- **Priorizar recall:** detecta más disputas (más alertas), a costa de más falsos positivos.  
- **Personalizado:** ajusta manualmente el umbral para tu operación (p. ej., 0.35).
"""
)
st.info(
"Ejemplo: si el umbral es 0.35 y la probabilidad predicha es 0.41, el caso se etiqueta como **Disputa**."
)

st.header("Campos del formulario")
st.markdown(
"""
- **Product / Sub-product:** tipo y subtipo del producto asociado a la queja.  
- **Issue / Sub-issue:** descripción categórica del problema.  
- **State / ZIP code:** localización del consumidor.  
- **Company / Company response:** empresa y tipo de respuesta dada.  
- **Date received / Date sent to company:** fechas para calcular **Days_to_response**.
"""
)

st.header("Cómo interpretamos la predicción")
st.markdown(
"""
- **Probabilidad**: riesgo estimado de disputa para el caso.  
- **Etiqueta**: *Disputa* / *No disputa* según el umbral activo.  
- **Explicación (SHAP)**: top factores que empujan la predicción hacia **más** o **menos** riesgo.
"""
)

st.header("Sugerencias de acción (ejemplos)")
st.markdown(
"""
- **Responder en <24h** cuando *Days_to_response* sea alto o el tipo de respuesta histórica eleve el riesgo.  
- **Escalar a agente sénior** si la probabilidad supera el umbral de alerta (p. ej., 0.7).  
- **Revisar plantillas de respuesta** para tipos que históricamente elevan el riesgo.
"""
)

st.header("Privacidad y uso responsable")
st.markdown(
"""
- El modelo no usa datos personales sensibles.  
- Las predicciones son **apoyo a la decisión**, no decisiones automáticas.  
- Mantén auditorías: guarda predicciones, umbrales y acciones realizadas.
"""
)

st.header("FAQs")
with st.expander("¿Por qué cambia la etiqueta si muevo el umbral?"):
    st.write("Porque el umbral define desde qué probabilidad consideras 'Disputa'. Un umbral más bajo etiqueta más casos como disputa.")
with st.expander("¿Puedo cargar el CSV original del cliente?"):
    st.write("Sí. En **Scoring masivo** la app detecta el formato original y lo normaliza automáticamente.")
with st.expander("¿Qué significa la importancia de una variable?"):
    st.write("Es el impacto promedio de esa variable en la salida del modelo. Con SHAP puedes ver su contribución caso a caso.")

st.divider()
st.caption("Proyecto Académico de ML – Predicción de disputas • Anthony Quiliche")
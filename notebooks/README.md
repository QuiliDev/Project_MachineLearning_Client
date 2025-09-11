# 01_Fuentes — Carga, unión y definición del problema

**Problema de ML**: Predecir si el cliente **disputará** la queja (`Consumer disputed?`) en el momento en que la queja **es recibida**.

**Unidad de predicción**: 1 fila = 1 queja (Complaint ID).

**Momento de predicción**: `Date received`.  
**Variables prohibidas por fuga** (ocurren después): `Date sent to company`, `Company response`, cualquier resultado posterior.

**Target**: `Consumer disputed?` (Yes/No) → binaria (1/0).

**Outputs de este notebook**:
- `/data/interim/quejas_raw.parquet`: dataset cargado con tipos correctos y checks básicos.
- Documento de **definición del problema** y **auditoría de fuga**.

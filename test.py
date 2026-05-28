from kgiant_distance_predictor import run_pipeline

result = run_pipeline(
    csv_path="data.csv",
    features=["logg","feh","g_r","e_logg","e_feh","e_g_r","teff","e_teff"],
    m_col="rmag",
    target_col="dm50"
)

print(f"R²: {result['r2']:.4f}")
print(f"RMSE: {result['rmse']:.4f}")
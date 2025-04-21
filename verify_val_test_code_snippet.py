# Code snippet to verify validation and test datasets images inside the notebook

# Assuming multi_val, multi_test, BalancedMultiLabelDamageDataset, get_transforms, verify_dataset are already defined in the notebook

# Convert string lists if needed
def convert_string_lists(df):
    import ast
    df['partes'] = df['partes'].apply(ast.literal_eval)
    df['dannos'] = df['dannos'].apply(ast.literal_eval)
    df['sugerencias'] = df['sugerencias'].apply(ast.literal_eval)
    return df

multi_val = convert_string_lists(multi_val)
multi_test = convert_string_lists(multi_test)

# Rename columns to match dataset class expectations
multi_val = multi_val[['Imagen', 'dannos', 'partes', 'sugerencias']].rename(columns={
    'dannos': 'damages',
    'partes': 'parts',
    'sugerencias': 'suggestions'
})

multi_test = multi_test[['Imagen', 'dannos', 'partes', 'sugerencias']].rename(columns={
    'dannos': 'damages',
    'partes': 'parts',
    'sugerencias': 'suggestions'
})

# Load transforms
data_transforms = get_transforms()

# Create datasets
val_dataset = BalancedMultiLabelDamageDataset(multi_val, 'data/fotos_siniestros/', data_transforms['val'])
test_dataset = BalancedMultiLabelDamageDataset(multi_test, 'data/fotos_siniestros/', data_transforms['test'])

# Verify datasets
print("Verifying validation dataset:")
verify_dataset(val_dataset)

print("Verifying test dataset:")
verify_dataset(test_dataset)

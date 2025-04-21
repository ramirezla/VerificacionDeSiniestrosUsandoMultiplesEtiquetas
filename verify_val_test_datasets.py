from DetectarDannosPartesSugerenciasUsandoMultiplesEtiquetas_V5 import BalancedMultiLabelDamageDataset, get_transforms, verify_dataset
import pandas as pd

# Load validation and test CSVs
multi_val = pd.read_csv('data/fotos_siniestros/datasets/multi_val.csv', sep='|')
multi_test = pd.read_csv('data/fotos_siniestros/datasets/multi_test.csv', sep='|')

# Convert string lists if needed (assuming function available)
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

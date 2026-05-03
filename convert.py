import tensorflow as tf
import os
import shutil

# Путь к исходной модели
input_model = 'classification.keras'

# Путь для сохранения (папка, без расширения!)
output_dir = 'animal_model_savedmodel'

# Удаляем папку если существует
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"🗑️  Удалена старая папка {output_dir}")

print(f"Загрузка модели из {input_model}...")
model = tf.keras.models.load_model(input_model, compile=False)

print(f"Конвертация в SavedModel формат...")
model.export(output_dir)

print(f"\n✅ Модель успешно конвертирована!")
print(f"📁 Папка: {output_dir}")
print(f"📄 Файлы:")
for root, dirs, files in os.walk(output_dir):
    level = root.replace(output_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f'{subindent}{file} ({size:.2f} MB)')
#!/usr/bin/env python3
import json
from huggingface_hub import HfApi

def main():
    api = HfApi()
    
    # Получаем список моделей с фильтрацией по text-to-image
    models = list(api.list_models(filter="text-to-image", sort="downloads", direction=-1))
    
    # Фильтруем только inpaint-модели (проверяем по названию или тегам)
    inpaint_models = [
        model for model in models 
        if "inpaint" in model.modelId.lower() or (model.tags and "inpainting" in model.tags)
    ]
    
    # Берем топ-5 моделей по загрузкам
    top5 = inpaint_models[:5]
    
    # Формируем словарь, где ключи - названия моделей, а значения - идентификаторы
    model_dict = {model.modelId.split("/")[-1]: model.modelId for model in top5}
    
    # Сохраняем результаты в JSON-файл
    with open("inpaint_models.json", "w", encoding="utf-8") as f:
        json.dump(model_dict, f, ensure_ascii=False, indent=4)
    
    print("Топ-5 inpaint diffusion моделей сохранены в inpaint_models.json")
    
if __name__ == "__main__":
    main()

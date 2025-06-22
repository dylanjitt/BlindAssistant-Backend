import os
from fastapi.testclient import TestClient
from src.api import app
import pytest

client = TestClient(app)

@pytest.mark.parametrize("image_name, expected_keywords", [
    ("IMG_20250321_171654979_HDR.jpg", ["auto", "rojo","piedra"]),
    ("IMG_20250402_070531071_HDR.jpg", ["Acetilcisteina","Acetilcisteína","COFAR"]),
    ("Imagen de WhatsApp 2025-05-28 a las 16.09.41_6653ed79.jpg", ["hiller","electric","amarillo","amarilla"]),
    ("IMG_20250522_100430496_HDR.jpg", ["Gato","Gatos","blanco","siames"]),
    ("IMG_20250523_222822732_HDR.jpg", ["BOLIVIANO","ACHUMANI","INFORMACIÓN","DE","PRESENTACIÓN","ministerio","defensa","bolivia"]),
    ("IMG_20250523_222930929_HDR.jpg", ["Dylan","Diego","Jitton","Zorrilla","Soltero","Estudiante"]),
    ("IMG_20250617_114422823_HDR.jpg", ["3,00", "tarifa","diurna","tramo","largo","3,20"]),
    ("IMG_20250621_213040367_HDR.jpg", ["mueble", "metal","dorado","mesa","madera","cortina"]),
    ("IMG_20250621_213053997_HDR.jpg", ["sonic","hedgehog","Switch","Shadow","Raros","unidos"]),
    ("IMG_20250621_213106787_HDR.jpg", ["DEEP","COOL", "geforce","rtx","componente","morado"])
])
def test_bolivian_money_detector(image_name, expected_keywords):
    image_path = os.path.join(os.path.dirname(__file__), "data","llm", image_name)

    with open(image_path, "rb") as f:
        response = client.post(
            "/ollamaVision",
            files={"file": (image_name, f, "image/jpeg")}
        )

    assert response.status_code == 200
    json_data = response.json()
    description = json_data.get("description", "").lower()

    assert any(word.lower() in description for word in expected_keywords), \
        f"None of the expected words {expected_keywords} were found in: {description}"


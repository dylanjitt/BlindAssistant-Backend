import os
from fastapi.testclient import TestClient
from src.api import app
import pytest

client = TestClient(app)

@pytest.mark.parametrize("image_name, expected_keywords", [
    ("test1.jpg", ["SAN PEDRO", "COLOMBIA","PRADO","OBELISCO","PEREZ V"]),
    ("test2.jpg", ["VITA", "UMSA","PRADO","OBRAJES","SAN MIGUEL"]),
    ("test3.jpg", ["PEREZ", "PRADO","UMSA","OBRAJES","CALACOTO"]),
    ("test4.jpg", ["UMSA", "SAN PEDRO","RODRIGUEZ"]),
    ("test5.jpg", ["ARCE", "SAN PEDRO","RODRIGUEZ","UMSA","TELEFERICO"]),
    ("test6.jpg", ["OBRAJES", "CALACOTO","SAN MIGUEL","LOS PINOS","PEDREGAL","ALMENDROS"]),\
    ("test7.jpg", ["ACHUMANI", "COMPLEJO","OBRAJES","CALACOTO","SAN MIGUEL"]),
    ("test8.jpg", ["OBRAJES","CALACOTO""MEGA CENTER","IRPAVI 2","BAJO IRPAVI"]),
    ("test9.jpg", ["ARCE", "UMSA","PRADO","P ESTUDIANTE","EST CENTRAL","TERMINAL"]),
    ("test10.jpg", ["ACHUMANI","OBRAJES","CALACOTO","SAN MIGUEL"]),
])
def test_minibus_sign_detector(image_name, expected_keywords):
    image_path = os.path.join(os.path.dirname(__file__), "data","minibus", image_name)

    with open(image_path, "rb") as f:
        response = client.post(
            "/minibusSignDetector",
            files={"file": (image_name, f, "image/jpeg")}
        )

    assert response.status_code == 200
    json_data = response.json()
    description = json_data.get("description", "").lower()

    assert any(word.lower() in description for word in expected_keywords), \
        f"None of the expected words {expected_keywords} were found in: {description}"


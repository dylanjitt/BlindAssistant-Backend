import os
from fastapi.testclient import TestClient
from src.api import app
import pytest

client = TestClient(app)

@pytest.mark.parametrize("image_name, expected_keywords", [
    ("testt.jpg", ["Bolivianos", "20","10","50"]),
    ("20.jpg", ["20"]),
    ("10.jpg", ["10"]),
    ("50.jpg", ["50"]),
    ("100.jpg", ["100"]),
    ("200.jpg", ["200"]),
    ("10y20.jpg", ["10", "20"]),
    ("20y10.jpg", ["20", "10"]),
    ("nada.jpg", ["No", "Banknotes"])
])
def test_bolivian_money_detector(image_name, expected_keywords):
    image_path = os.path.join(os.path.dirname(__file__), "data","billetes", image_name)

    with open(image_path, "rb") as f:
        response = client.post(
            "/BolivianMoneyDetector",
            files={"file": (image_name, f, "image/jpeg")}
        )

    assert response.status_code == 200
    json_data = response.json()
    description = json_data.get("description", "").lower()

    assert any(word.lower() in description for word in expected_keywords), \
        f"None of the expected words {expected_keywords} were found in: {description}"


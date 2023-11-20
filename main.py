import os
from requests import request
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from pydantic import BaseModel
from urllib.request import urlopen
import urllib.parse
from fastapi import FastAPI, Response, UploadFile, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import tensorflow as tf
import json

app = FastAPI()

categories = ['bawangBombay',
 'bayamHijau',
 'bayamMerah',
 'belut',
 'buahAlpukat',
 'buahAnggur',
 'buahApel',
 'buahBelimbing',
 'buahDuku',
 'buahDurian',
 'buahJambuAir',
 'buahJambuBiji',
 'buahJeruk',
 'buahJerukBali',
 'buahJerukNipis',
 'buahKedondong',
 'buahKelapa',
 'buahKelengkeng',
 'buahKesemek',
 'buahLemon',
 'buahMangga',
 'buahManggis',
 'buahMarkisa',
 'buahMatoa',
 'buahMelon',
 'buahMengkudu',
 'buahNaga',
 'buahNanas',
 'buahNangka',
 'buahPala',
 'buahPepaya',
 'buahPisang',
 'buahRambutan',
 'buahSalak',
 'buahSawo',
 'buahSemangka',
 'buahSirsak',
 'buahSrikaya',
 'buahSukun',
 'cumiCumi',
 'dagingAyam',
 'dagingBebek',
 'dagingKambing',
 'dagingSapi',
 'daunKatuk',
 'daunKubis',
 'daunKucai',
 'daunMelinjo',
 'daunParsley',
 'daunPepaya',
 'daunSeledri',
 'daunSingkong',
 'daunTalas',
 'ecengGondok',
 'ikanBandeng',
 'ikanBawal',
 'ikanBili',
 'ikanBubara',
 'ikanCakalang',
 'ikanEkorKuning',
 'ikanGabus',
 'ikanKakapMerah',
 'ikanKembung',
 'ikanLele',
 'ikanMas',
 'ikanMujair',
 'ikanNila',
 'ikanPatin',
 'ikanPindang',
 'ikanSalmon',
 'ikanSarden',
 'ikanSepat',
 'ikanTeri',
 'ikanTongkol',
 'ikanTuna',
 'jamurKuping',
 'jamurTiram',
 'jengkol',
 'kacangMekah',
 'kecombrang',
 'kepiting',
 'kerang',
 'labuWaluhKuning',
 'melinjo',
 'petai',
 'rebung',
 'rumputLaut',
 'sayurBuncis',
 'sayurGenjer',
 'sayurJagungMuda',
 'sayurJantungPisang',
 'sayurKacangPanjang',
 'sayurKangkung',
 'sayurKecipir',
 'sayurKolMerah',
 'sayurKolPutih',
 'sayurLabuSiam',
 'sayurNangkaMuda',
 'sayurPakis',
 'sayurPepayaMuda',
 'sayurSawiHijau',
 'sayurSawiPutih',
 'sayurSelada',
 'sayurSeladaAir',
 'sayurTerong',
 'sayurTerongBelanda',
 'taoge',
 'timun',
 'tomatMerah',
 'tomatMuda',
 'udangGalah',
 'udangRebon',
 'udangWindu',
 'wortel'
 ]
model = tf.keras.models.load_model("models.h5")

def load_json_data(filename):
    with open(filename, "r") as json_file:
        return json.load(json_file)

kdg = "dataKandunganGizi.json"
mft = "manfaatMakanan.json"
kandunganGizi = load_json_data(kdg)
manfaatMakanan = load_json_data(mft)

@app.get("/")
async def home():
    return {"Successfully": "Deployed"}


@app.post("/predict_image")
def predict_image(request: Request, url: str, response: Response):
    try:
        encoded_url = urllib.parse.quote(url, safe=":/")
        image_response = urlopen(encoded_url)
        if image_response.getcode() != 200:
            response.status_code = 400
            return "Failed to retrieve image from URL!"

        image_data = image_response.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((384, 384))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.cast(tf.expand_dims(img_array, 0), tf.float32)
        images = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        result = model.predict(images, batch_size=32)
        
        gizi = None
        manfaat = None
        indexing_for_json = str(np.argmax(result).item())

        if indexing_for_json in kandunganGizi:
            gizi = kandunganGizi[indexing_for_json]
            print(f"{gizi}")

        if indexing_for_json in manfaatMakanan:
            manfaat = manfaatMakanan[indexing_for_json]
            print(f"{manfaat}")

        return {
        "index": np.argmax(result).item(),
        "result": categories[np.argmax(result)],
        "akurasi": np.round(100 * np.max(result), 2),
        "gizi": gizi,
        "manfaat": manfaat
        }

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {str(e)}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host="0.0.0.0", port=int(port))

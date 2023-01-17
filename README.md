# Virhe-api

## Asennus

- Luo ja aktivoi conda-ympäristö:

`conda create -n virhe_api_env python=3.7`

`conda activate virhe_api_env`

- Asenna poppler:

`conda install -c conda-forge poppler`

- Asenna muut riippuvuudet:

`pip install -r requirements.txt`

## Yleistä

- Käynnistys: 

`flask --app api.py run`

- Käynnistys debuggauksella: 

`flask --app api.py --debug run`

- API olettaa että esikoulutetut mallit löytyvät kansiosta './mallit', ja niiden nimet ovat
'post_it_model_20122022.onnx', 'corner_model_19122022.onnx', 'empty_model_v4.onnx'.

Mallit on tallennettu koneoppimismalleille luodussa avoimessa onnx-formaatissa (https://onnx.ai/), mikä nopeuttaa niiden toimintaa ja tekee ne riippumattomiksi mallien kouluttamiseen käytetystä kirjastosta.

- Yksi route: '/detect'

- Komponenttivalinnat ilmaistaan muodossa `postit=1&corner=1&empty=1`, jolloin
POST-pyynnön url on muotoa `/detect?postit=1&corner=1&empty=1`

- Default-portti: 5000

- Odottaa että input-kuva/tiedosto löytyy POST-requestista 'image'-nimisellä avaimella (`request.files["image"]`)

- Palauttaa Flaskin Responsen, jossa `'Content-Type': 'text/csv'` ja tulokset ovat .csv-tiedostossa nimeltä 'virheet.csv'

- Toimintaa voi testata oheisen request.py-koodin avulla (tai Postmanilla, curlilla tms.)

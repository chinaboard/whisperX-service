# WhisperX Service

This is an API service that receives audio file paths via the endpoint `POST /asr`.

For the server specification (request structure and response behavior) see the OpenAPI specificaiton in `/docs`.

For any other documentation refer to [WhisperX readme](https://github.com/m-bain/whisperX).

## Features
Current release (v1.0.0) supports following whisper models:

- [whisperX](https://github.com/m-bain/whisperX)@[v3.1.1](https://github.com/m-bain/whisperX/releases/tag/v3.1.1)

![Swagger UI](https://github.com/chinaboard/whisperX-service/blob/master/docs/assets/img/swagger-ui.png?raw=true)

Available ASR_MODELs are `tiny`, `base`, `small`, `medium`, `large`, `large-v1` and `large-v2`. Please note that `large` and `large-v2` are the same model.

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

## Quick start

After running the docker image interactive Swagger API documentation is available at [localhost:9000/docs](http://localhost:9000/docs)

There are 2 endpoints available:

- /asr (TXT, VTT, SRT, TSV, JSON)
- /detect-language

## Docker Build

### For GPU

```sh
# Build Image
docker build -t whisperx-service .

# Run Container
docker run -d --gpus all -p 9000:9000 whisperx-service
# or
docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base whisperx-service
```

## Cache
The ASR model is downloaded each time you start the container, using the large model this can take some time. If you want to decrease the time it takes to start your container by skipping the download, you can store the cache directory (/root/.cache/whisper) to an persistent storage. Next time you start your container the ASR Model will be taken from the cache instead of being downloaded again.

**Important this will prevent you from receiving any updates to the models.**
 
```sh
docker run -d -p 9000:9000 -e ASR_MODEL=large -v /tmp/whisper:/root/.cache/whisper whisperx-service
```


## Related
- [ahmetoner/whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)
- [alexgo84/whisperx-server](https://github.com/alexgo84/whisperx-server)
import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch
import whisper
import whisperx
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON

model_name= os.getenv("ASR_MODEL", "large")
x_models = dict()

if torch.cuda.is_available():
    device = "cuda"
    model = whisper.load_model(model_name).cuda()
else:
    print('torch cuda is not available')
    sys.exit(1)
model_lock = Lock()

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    output
):
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    with model_lock:
        result = model.transcribe(audio, **options_dict)

    # Load the required model and cache it
    # If we transcribe models in many differen languages, this may lead to OOM propblems
    if result["language"] in x_models:
        print('Using chached model')
        model_x, metadata = x_models[result["language"]]
    else:
        print(f'Loading model {result["language"]}')
        x_models[result["language"]] = whisperx.load_align_model(language_code=result["language"], device=device)
        model_x, metadata = x_models[result["language"]]

    # Align whisper output
    result = whisperx.align(result["segments"], model_x, metadata, audio, device, return_char_alignments=False)

    outputFile = StringIO()
    write_result(result, outputFile, output)
    outputFile.seek(0)

    return outputFile

def language_detection(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)

    return detected_lang_code

def write_result(
    result: dict, file: BinaryIO, output: Union[str, None]
):
    if(output == "srt"):
        WriteSRT(ResultWriter).write_result(result, file, {})
    elif(output == "vtt"):
        WriteVTT(ResultWriter).write_result(result, file, {})
    elif(output == "tsv"):
        WriteTSV(ResultWriter).write_result(result, file, {})
    elif(output == "json"):
        WriteJSON(ResultWriter).write_result(result, file, {})
    elif(output == "txt"):
        WriteTXT(ResultWriter).write_result(result, file, {})
    else:
        return 'Please select an output method!'

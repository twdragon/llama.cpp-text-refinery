#!/usr/bin/env python3
import sys
import requests
import json
from pathlib import Path
import re
from math import ceil
import numpy
import logging
import atexit
import subprocess
import platform
import time
import yaml
import argparse


def stop_http_llama_server(server_process, logger):
    try:
        logger.info('Terminating server')
        server_process.terminate()
        server_process.wait(timeout=5)
    except Exception as e:
        logger.error('Error stopping server: {}'.format(str(e)))
        server_process.kill()


def remove_ffmpeg_artifacts(ffmpeg_output, whisper_output, logger, reuse_txt):
    logger.info('Removing artifacts\n\t{}\n\t{}'.format(str(ffmpeg_output), str(whisper_output)))
    ffmpeg_output.unlink(missing_ok=True)
    if not reuse_txt:
        whisper_output.unlink(missing_ok=True)


log = logging.getLogger('llama-refinery')
logging.basicConfig(level=logging.INFO)

# Arguments handler
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIRECTORY.joinpath('config.yml')
argument_parser = argparse.ArgumentParser(description='llama.cpp configurable text refinery by twdragon')
argument_parser.add_argument('input_text',
                             action='store', 
                             help='File containing input text for refining',
                             type=str)
argument_parser.add_argument('-es', '--external-server', 
                             action='store_false', 
                             help='Use external llama.cpp server instead of running our own')
argument_parser.add_argument('-c', '--config', 
                             action='store', 
                             default=str(DEFAULT_CONFIG),
                             const=str(DEFAULT_CONFIG),
                             nargs='?',
                             type=str,
                             help='YAML configuration file, the default is {}'.format(str(DEFAULT_CONFIG)))
argument_parser.add_argument('-m', '--model', 
                             action='store', 
                             default=None,
                             const=None,
                             nargs='?',
                             type=str,
                             help='Quantized LLM file to load, overrides config file setting')
argument_parser.add_argument('-cl', '--context-length', 
                             action='store', 
                             default=4096,
                             const=4096,
                             nargs='?',
                             type=int,
                             help='LLM context window length limit [4096]')
argument_parser.add_argument('-b', '--batch-length', 
                             action='store', 
                             default=2048,
                             const=2048,
                             nargs='?',
                             type=int,
                             help='LLM loader batch length [2048]')
argument_parser.add_argument('-w', '--window-coeff', 
                             action='store', 
                             type=float,
                             default=None,
                             const=None,
                             nargs='?',
                             help='Context window width coefficient to calculate the tokenizer context window, window-width = window-coeff * context-length [0.6714], overrides config file setting')
argument_parser.add_argument('-ov', '--overlap-coeff', 
                             action='store', 
                             type=float,
                             default=None,
                             const=None,
                             nargs='?',
                             help='Context window overlap coefficient [0.05], overrides config file setting')
argument_parser.add_argument('-pl', '--predict-limit-coeff', 
                             action='store', 
                             type=float,
                             default=None,
                             const=None,
                             nargs='?',
                             help='Prediction limit calculation coefficient [0.4272], overrides config file setting')
argument_parser.add_argument('--model-remap', 
                             action='store_false', 
                             help='Set on model RAM remapping mode (for multiple-instance or multi-GPU servers)')
argument_parser.add_argument('--enable-webui', 
                             action='store_false', 
                             help='Turn on web UI renderer on llama.cpp server')
argument_parser.add_argument('-gl', '--gpu-layers', 
                             action='store', 
                             default=None,
                             const=None,
                             nargs='?',
                             type=int,
                             help='Instructs the server to upload a number of LLM layers to GPU VRAM [40]')
argument_parser.add_argument('-p', '--prompt-preset', 
                             action='store', 
                             default=None,
                             const=None,
                             nargs='?',
                             type=str,
                             help='Select the prompt preset instead of the default (first appeared) one')
argument_parser.add_argument('-wh', '--whisper', 
                             action='store_true', 
                             help='Use whisper.cpp utility to recognize speech. Requires whisper.cpp and ffmpeg being installed')
argument_parser.add_argument('-wl', '--whisper-language', 
                             action='store', 
                             default=None,
                             const=None,
                             nargs='?',
                             type=str,
                             help='Select the language preset for whisper.cpp')
argument_parser.add_argument('-wt', '--whisper-keep-txt', 
                             action='store_true', 
                             help='Do not remove TXT artifact from whisper.cpp')

arguments = argument_parser.parse_args()

# Configuration loader
CONFIG_FILE = Path(arguments.config)
if not CONFIG_FILE.is_file():
    log.error('Configuration file {} not found, falling back to the default one {}'.format(arguments.config, str(DEFAULT_CONFIG)))
    CONFIG_FILE = DEFAULT_CONFIG
config_handler = CONFIG_FILE.open('tr', encoding='utf-8')
config = yaml.safe_load(config_handler)
config_handler.close()

if not Path(arguments.input_text).is_file():
    log.error('Input file \'{}\' not found!'.format(str(arguments.input_text)))
    exit(1)

TRANSCRIPT_FILE = Path(arguments.input_text).resolve()

# Running Whisper if needed
if arguments.whisper:
    log.info('Extended speech recognition mode selected, treating the input file as multimedia')
    log.info('Running audio processing')
    ffmpeg_output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem) + '.wav')
    whisper_output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem) + '.txt')
    atexit.register(remove_ffmpeg_artifacts, ffmpeg_output_filename, whisper_output_filename, log, arguments.whisper_keep_txt)
    FFMPEG_EXECUTABLE = config['whisper']['ffmpeg'][platform.system()]
    if not Path(FFMPEG_EXECUTABLE).is_file():
        log.error('FFMPEG executable file \'{}\' not found!'.format(FFMPEG_EXECUTABLE))
        exit(1)
    ffmpeg_cli_list = [FFMPEG_EXECUTABLE,
                       '-i',
                       str(TRANSCRIPT_FILE),
                       '-vn',
                       '-c:a',
                       'pcm_s16le',
                       '-ar',
                       '16000',
                       '-ac',
                       '1',
                       '-y',
                       '-f',
                       'wav',
                       str(ffmpeg_output_filename)]
    try:
        subprocess.run(ffmpeg_cli_list, check=True)
    except Exception as e:
        logger.error('Error processing audio: {}'.format(str(e)))
        exit(1)
    log.info('Running speech recognition')
    WHISPER_EXECUTABLE = config['whisper']['executable'][platform.system()]
    if not Path(WHISPER_EXECUTABLE).is_file():
        log.error('whisper.cpp executable file \'{}\' not found!'.format(WHISPER_EXECUTABLE))
        exit(1)
    WHISPER_MODEL_FILENAME = config['whisper']['model_filename']
    WHISPER_MODEL_PATH = Path(config['whisper']['model_dir'][platform.system()])
    WHISPER_MODEL_PATH = WHISPER_MODEL_PATH.joinpath(WHISPER_MODEL_FILENAME).resolve()
    if not WHISPER_MODEL_PATH.is_file():
        log.error('Whisper model file \'{}\' not found!'.format(str(WHISPER_MODEL_PATH)))
        exit(1)
    whisper_cli_list = [WHISPER_EXECUTABLE,
                        '-m',
                        str(WHISPER_MODEL_PATH),
                        '-mc',
                        '4',
                        '-otxt',
                        '-l',
                        'auto' if arguments.whisper_language is None else str(arguments.whisper_language),
                        '-f',
                        str(ffmpeg_output_filename),
                        '-of',
                        TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem))]
    try:
        subprocess.run(whisper_cli_list, check=True)
    except Exception as e:
        logger.error('Error processing audio: {}'.format(str(e)))
        exit(1)
    TRANSCRIPT_FILE = whisper_output_filename.resolve()

LLAMASERVER_EXECUTABLE = config['server_executable'][platform.system()]
LLAMASERVER_MODEL_FILENAME = config['llm_filename']
LLAMASERVER_MODEL_PATH = Path(config['llm_dir'][platform.system()])
LLAMASERVER_MODEL_PATH = Path(arguments.model).resolve() if arguments.model is not None else LLAMASERVER_MODEL_PATH.joinpath(LLAMASERVER_MODEL_FILENAME).resolve()

LLAMASERVER_URI = 'http://' if not config['llama_server_https'] else 'https://'
LLAMASERVER_URI += config['llama_server_host'] + ':' + str(config['llama_server_port']) + '/'
LLAMASERVER_GENERATION_ENDPOINT = LLAMASERVER_URI + config['llama_server_generation_endpoint']
LLAMASERVER_TOKENIZER_ENDPOINT = LLAMASERVER_URI + config['llama_server_tokenizer_endpoint']
LLAMASERVER_CHECKUP_ENDPOINT = LLAMASERVER_URI + config['llama_server_diagnostic_endpoint']

if arguments.external_server:
    log.info('Running server')
    if not Path(LLAMASERVER_EXECUTABLE).is_file():
        log.error('llama.cpp server executable file \'{}\' not found!'.format(str(LLAMASERVER_EXECUTABLE)))
        exit(1)
    if not LLAMASERVER_MODEL_PATH.is_file():
    log.error('Model file \'{}\' not found!'.format(str(LLAMASERVER_MODEL_PATH)))
    exit(1)
    server_cli_list = [LLAMASERVER_EXECUTABLE, 
                       '--model',
                       str(LLAMASERVER_MODEL_PATH),
                       '-c',
                       str(arguments.context_length),
                       '-b',
                       str(arguments.batch_length)]
    if arguments.model_remap:
        server_cli_list.append('--no-mmap')
    if arguments.enable_webui:
        server_cli_list.append('--no-webui')
    if arguments.gpu_layers is not None:
        server_cli_list.append('--gpu-layers')
        server_cli_list.append(str(arguments.gpu_layers))
    server_process = subprocess.Popen(server_cli_list)
    time.sleep(2)
    atexit.register(stop_http_llama_server, server_process, log)
else:
    log.info('User setting: EXTERNAL_SERVER, not running llama.cpp server explicitly')

log.info('Querying model load, 60 sec timeout. Please wait')
error_count = 0
server_ready = False
while error_count < 30:
    time.sleep(2)
    with requests.get(LLAMASERVER_CHECKUP_ENDPOINT) as response:
        if response.status_code != 200:
            error_count += 1
        else:
            server_ready = True
            break

if not server_ready:
    log.error('Server is not ready. Operation aborted')
    exit(1)

WINDOW_WIDTH = ceil(config['window_width_coefficient'] * arguments.context_length) if arguments.window_coeff is None else ceil(arguments.window_coeff * arguments.context_length)
WINDOW_OVERLAP_COEFF = config['window_overlap_coefficient'] if arguments.overlap_coeff is None else arguments.overlap_coeff
window_limit = ceil(WINDOW_WIDTH * (1 - WINDOW_OVERLAP_COEFF))
log.info('Calculating rolling window parameters: {} tokens of {} in context, every {} tokens in the text'. format(str(WINDOW_WIDTH), arguments.context_length, str(window_limit)))
PREDICTION_LIMIT = ceil(config['prediction_limit_coefficient'] * arguments.context_length) if arguments.predict_limit_coeff is None else ceil(arguments.predict_limit_coeff * arguments.context_length)
log.info('Running tokenizer')

sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s*(?:\n\s*)*')
inline_pattern = re.compile(r'(?<![\.\?\!])\n+') 
alphanumeric_pattern = re.compile(r'[^0-9a-zA-Z]+')
transcript_handle = TRANSCRIPT_FILE.open('tr', encoding='utf-8')
transcript = transcript_handle.read()
transcript = re.sub(inline_pattern, ' ', transcript)
sentences = re.split(sentence_pattern, transcript)
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
total_sentences = len(sentences)
transcript_handle.close()

sentence_borders = list()
total_tokens = 0
for sentence in sentences:
    payload = {
        'content': sentence,
        'add_special': False,
        'with_pieces': False
    }
    with requests.post(LLAMASERVER_TOKENIZER_ENDPOINT, json=payload, stream=False) as response:
        if response.status_code == 200:
            tokens = json.loads(response.text)['tokens']
            total_tokens += len(tokens)
            sentence_borders.append(total_tokens)
        else:
            log.error('HTTP error: status code {} in tokenizer'.format(response.status_code))
assert(len(sentences) == len(sentence_borders))
log.info('Running on {} sentences'.format(len(sentence_borders)))

batched = False
batches = list()

if sentence_borders[-1] > WINDOW_WIDTH: 
    batched = True
    log.info('Preparing batches')
    current_position = 0
    current_start_token = 0
    while current_start_token < sentence_borders[-1]:
        end_position = (numpy.abs(numpy.asarray(sentence_borders) - (current_start_token + WINDOW_WIDTH) )).argmin()
        next_start_position = (numpy.abs(numpy.asarray(sentence_borders) - (current_start_token + window_limit) )).argmin()
        if end_position >= ceil(len(sentences) * (1 - WINDOW_OVERLAP_COEFF)):
            batches.append(sentences[current_position:])
            break
        else:
            batches.append(sentences[current_position:end_position])
        current_start_token += window_limit
        current_position = next_start_position
    log.info('Processed {} batches'.format(len(batches)))
else:
    log.info('No batches required')
log.info('Loading prompts')
if not isinstance(config['presets'], list):
    log.error('Invalid configuration! Please ensure that the presets list is an iterable entity!')
    exit(1)
presets = config['presets']
default_preset = presets[0]
preset = default_preset

if arguments.prompt_preset is not None:
    try:
        preset = next(p for p in presets if p['name'] == arguments.prompt_preset or p['cli_alias'] == arguments.prompt_preset)
    except StopIteration:
        preset = None

if preset is None:
    log.warning('Selected preset {} not found, falling back to the default one {}'.format(arguments.prompt_preset, default_preset['name']))
    preset = default_preset

log.info('Running processing')
responses = dict()
for prompt in preset['prompts']:
    responses[prompt['name'].lower()] = str()

if batched:
    for i, batch in enumerate(batches):
        log.info('Processing batch {} of {}'.format(i + 1, len(batches)))
        for prompt in preset['prompts']:
            payload = dict()
            prompt_payload = str()
            prompt_payload += prompt['text_before'] if 'text_before' in prompt.keys() else str()
            prompt_payload += ' '.join(batch) + '\n\n'
            prompt_payload += prompt['text_after'] if 'text_after' in prompt.keys() else str()
            model_name = str(Path(config['llm_filename']).stem)
            if model_name in config:
                for param, val in config[model_name].items():
                    payload[param] = val
            payload['prompt'] = prompt_payload
            payload['stream'] = True
            payload['n_predict'] = PREDICTION_LIMIT
            with requests.post(LLAMASERVER_GENERATION_ENDPOINT, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            token = line.decode('utf-8')[6:]
                            json_object = json.loads(token)
                            content = json_object.get("content", "")
                            responses[prompt['name'].lower()] += content
                            # print('\rAcquired {} tokens'.format(response_tokens), end='', flush=True)
                            print(content, end='', flush=True)
                    print()
                    responses[prompt['name'].lower()] += '\n\n'
                else:
                    log.error('HTTP error: status code {}'.format(response.status_code))
else:
    log.info('Processing entire text')
    for prompt in preset['prompts']:
        payload = dict()
        prompt_payload = str()
        prompt_payload += prompt['text_before'] if 'text_before' in prompt.keys() else str()
        prompt_payload += ' '.join(sentences) + '\n\n'
        prompt_payload += prompt['text_after'] if 'text_after' in prompt.keys() else str()
        model_name = str(Path(config['llm_filename']).stem)
        if model_name in config:
            for param, val in config[model_name].items():
                payload[param] = val
        payload['prompt'] = prompt_payload
        payload['stream'] = True
        payload['n_predict'] = PREDICTION_LIMIT
        with requests.post(LLAMASERVER_GENERATION_ENDPOINT, json=payload, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        token = line.decode('utf-8')[6:]
                        json_object = json.loads(token)
                        content = json_object.get("content", "")
                        responses[prompt['name'].lower()] += content
                        # print('\rAcquired {} tokens'.format(response_tokens), end='', flush=True)
                        print(content, end='', flush=True)
                print()
                responses[prompt['name'].lower()] += '\n\n'
            else:
                log.error('HTTP error: status code {}'.format(response.status_code))

log.info('Saving results')
for partname, mdtext in responses.items():
    output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem + '.' + re.sub(alphanumeric_pattern, '_', partname) + '.md'))
    output_handle = output_filename.open('wt', encoding='utf-8')
    output_handle.write(mdtext)
    output_handle.close()
    log.info('Output for the prompt {} is saved to {}'.format(partname, str(output_filename)))

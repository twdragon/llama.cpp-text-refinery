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
import hashlib


def stop_http_llama_server(server_process, logger):
    try:
        logger.info('Terminating server')
        server_process.terminate()
        server_process.wait(timeout=5)
    except Exception as e:
        logger.error('Error stopping server: {}'.format(str(e)))
        server_process.kill()


def remove_ffmpeg_artifacts(ffmpeg_output, whisper_output, logger, reuse_txt):
    info_string = 'Removing artifacts\n\t{}\n\t{}'.format(str(ffmpeg_output), str(whisper_output)) if not reuse_txt else 'Removing artifact\n\t{}'.format(str(ffmpeg_output))
    logger.info(info_string)
    ffmpeg_output.unlink(missing_ok=True)
    if not reuse_txt:
        whisper_output.unlink(missing_ok=True)


def render_markdown(heading, responses, filename, integrate_thumbnails=None):
    import markdown
    md_extensions = ['extra',
                     'toc',
                     'codehilite',
                     'markdown_katex',
                     'abbr',
                     'sane_lists']
    markdown_doc = '# {}\n\n'.format(heading)
    if integrate_thumbnails is not None:
        cnt = 1
        for thumbnail, frame in integrate_thumbnails:
            markdown_doc += '[ ![Sceencap]({}) ]({}) '.format(thumbnail, frame)
            markdown_doc += '\n\n' if (cnt % 4) == 0 else ''
            cnt +=1
        markdown_doc += '\n\n-----\n\n'
    for partname, mdtext in responses.items():
        markdown_doc +='## {}\n\n'.format(partname)
        markdown_doc += mdtext
        markdown_doc += '\n\n'
    html_doc = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=85%device-width, initial-scale=1.0">
    <meta name="description" content="application/xhtml+xml"/>
    <meta charset="UTF-8"/>
    <style>
        /* GitLab-like Markdown Styling by ChatGPT 4o */
        body {{
            font-family: "Times New Roman", Times, serif;
            font-size: 14pt;
            line-height: 1.6;
            color: #333;
            background-color: #fff;
            margin: 20px;
            padding: 20px;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-weight: bold;
            color: #24292e;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }}

        h1 {{ font-size: 24pt; border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }}
        h2 {{ font-size: 22pt; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h3 {{ font-size: 20pt; }}
        h4 {{ font-size: 18pt; }}
        h5 {{ font-size: 16pt; }}
        h6 {{ font-size: 14pt; color: #6a737d; }}

        p {{
            margin-bottom: 1em;
            font-size: 14pt;
        }}

        ul, ol {{
            padding-left: 2em;
            font-size: 14pt;
        }}

        li {{
            margin-bottom: 0.3em;
        }}

        pre {{
            background: #f6f8fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 11pt;
        }}

        code {{
            font-family: monospace;
            background: #f6f8fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 12pt;
        }}

        pre code {{
            display: block;
            padding: 10px;
        }}

        blockquote {{
            margin: 0;
            padding: 0.5em 1em;
            color: #6a737d;
            border-left: 4px solid #dfe2e5;
            background: #f8f9fa;
            font-size: 14pt;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1em;
            font-size: 14pt;
        }}

        table, th, td {{
            border: 1px solid #dfe2e5;
        }}

        th, td {{
            padding: 8px;
            text-align: left;
        }}

        th {{
            background: #f6f8fa;
            font-weight: bold;
        }}

        a {{
            color: #0366d6;
            text-decoration: none;
            font-size: 14pt;
        }}

        a:hover {{
            text-decoration: underline;
        }}
    </style>
    <title>{doctitle}</title>
</head>
<body>
{docrender}
</body>
</html>
'''
    markdown_doc = re.sub(r'\n{3,}', '\n\n', markdown_doc.strip())
    list_pattern = re.compile(r'(\s*[-*+]\s+|\s*\d+\.\s+)(.*)\n\n')
    markdown_doc = re.sub(list_pattern, r'\1\2\n', markdown_doc)
    html_doc = html_doc.format(doctitle=heading, docrender=markdown.markdown(markdown_doc, extensions=md_extensions))
    html_handler = filename.open('wt', encoding='utf-8')
    html_handler.write(html_doc)
    html_handler.close()


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
argument_parser.add_argument('-j', '--join-text', 
                             action='store_true', 
                             help='Join all the generated text into one Markdown document')
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
argument_parser.add_argument('-rm', '--render-markdown', 
                             action='store_true', 
                             help='Render Markdown to HTML (requires Python Markdown extension package installed)')
argument_parser.add_argument('--video-frames', 
                             action='store_true', 
                             help='Tries to extract 16 preview frames from the input video when rendering Markdown to HTML')

# Argument preservation chain
arguments = argument_parser.parse_args()
keep_text = arguments.whisper_keep_txt

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
video_frames = None
if arguments.whisper:
    log.info('Extended speech recognition mode selected, treating the input file as multimedia')
    if arguments.video_frames:
        log.info('Extended video speech recognition mode selected, rendering frames into thumbnails')
        video_duration = 0
        ffprobe_cli_list = ['ffprobe',
                            '-i',
                            str(TRANSCRIPT_FILE),
                            '-show_entries',
                            'format=duration',
                            '-v',
                            'quiet',
                            '-of',
                            'csv=p=0']
        try:
            ffprobe_handler = subprocess.run(ffprobe_cli_list, check=True, capture_output=True)
            video_duration = float(ffprobe_handler.stdout)
        except Exception as e:
            log.error('Error processing video: {}'.format(str(e)))
            exit(1)
        if video_duration < 180:
            log.warning('The estimated duration is less than 3 minutes, dropping video')
        else:
            hasher = hashlib.sha1()
            hasher.update(str(TRANSCRIPT_FILE).encode('utf-8'))
            transcript_hash = hasher.hexdigest()
            images_dir = TRANSCRIPT_FILE.parent.joinpath('img')
            images_dir.mkdir(parents=False, exist_ok=True)
            frames_generation_cli_list = ['ffmpeg',
                                          '-i',
                                          str(TRANSCRIPT_FILE),
                                          '-ss',
                                          '00:01:00',
                                          '-vf',
                                          'fps=16/({} - 120):round=down'.format(str(video_duration)),
                                          '-qscale:v',
                                          '8',
                                          '-fps_mode',
                                          'vfr',
                                          str(images_dir.joinpath(transcript_hash)) + '_full_%02d.jpg']
            thumbs_generation_cli_list = ['ffmpeg',
                                          '-i',
                                          str(TRANSCRIPT_FILE),
                                          '-ss',
                                          '00:01:00',
                                          '-vf',
                                          'fps=16/({} - 120):round=down,scale=300:-1'.format(str(video_duration)),
                                          '-qscale:v',
                                          '8',
                                          '-fps_mode',
                                          'vfr',
                                          str(images_dir.joinpath(transcript_hash)) + '_thumb_%02d.jpg']
            try:
                log.info('Generating images')
                subprocess.run(frames_generation_cli_list, check=True)
                log.info('Generating frames')
                subprocess.run(thumbs_generation_cli_list, check=True)
            except Exception as e:
                log.error('Error processing video: {}'.format(str(e)))
                exit(1)
            video_frames = list()
            for i in range(1, 17, 1):
                video_frames.append( ('./img/{}_thumb_{:02d}.jpg'.format(transcript_hash, i), './img/{}_full_{:02d}.jpg'.format(transcript_hash, i)) )
            log.info('Frame previews generated in {}'.format(str(images_dir)))
    log.info('Running audio processing')
    ffmpeg_output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem) + '.wav')
    whisper_output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem) + '.txt')
    atexit.register(remove_ffmpeg_artifacts, ffmpeg_output_filename, whisper_output_filename, log, keep_text)
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
    responses[prompt['name']] = str()

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
                            responses[prompt['name']] += content
                            # print('\rAcquired {} tokens'.format(response_tokens), end='', flush=True)
                            print(content, end='', flush=True)
                    print()
                    responses[prompt['name']] += '\n\n'
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
                        responses[prompt['name']] += content
                        # print('\rAcquired {} tokens'.format(response_tokens), end='', flush=True)
                        print(content, end='', flush=True)
                print()
                responses[prompt['name']] += '\n\n'
            else:
                log.error('HTTP error: status code {}'.format(response.status_code))

log.info('Saving results')
if arguments.render_markdown:
    output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem + '.html'))
    log.info('Rendering HTML file {}'.format(output_filename))
    render_markdown(str(TRANSCRIPT_FILE.stem),
                    responses,
                    output_filename,
                    video_frames)
elif arguments.join_text:
    output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem + '.md'))
    output_handle = output_filename.open('wt', encoding='utf-8')
    output_handle.write('# {}\n\n'.format(str(TRANSCRIPT_FILE.stem)))
    for partname, mdtext in responses.items():
        output_handle.write('## {}\n\n'.format(partname))
        output_handle.write(mdtext)
        output_handle.write('\n')
    output_handle.close()
    log.info('Full output saved to {}'.format(str(output_filename)))
else:
    for partname, mdtext in responses.items():
        output_filename = TRANSCRIPT_FILE.parent.joinpath(str(TRANSCRIPT_FILE.stem + '.' + re.sub(alphanumeric_pattern, '_', partname.lower()) + '.md'))
        output_handle = output_filename.open('wt', encoding='utf-8')
        output_handle.write(mdtext)
        output_handle.close()
        log.info('Output for the prompt {} is saved to {}'.format(partname, str(output_filename)))


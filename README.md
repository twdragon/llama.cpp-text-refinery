# [`llama.cpp`](https://github.com/ggml-org/llama.cpp) Configurable Text Refinery

Very simple configurable example of a prompt processing script to perform [almost] seamless summarization of the meeting transcripts.

## Installation

The script requires accesible executables for [`llama.cpp`](https://github.com/ggml-org/llama.cpp), optionally for [`ffmpeg`](https://www.ffmpeg.org/) and [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp). Intended to be cross-platform, it was tested on Ubuntu Linux, MacOS and Windows equipped with [Python 3](https://python.org).

## Documentation

The script supports the following command line options:

```sh
positional arguments:
  input_text            File containing input text/multimedia for refining

options:
  -h, --help            show this help message and exit
  -es, --external-server
                        Use external llama.cpp server instead of running our own
  -o, --output-dir [OUTPUT_DIR]
                        Output directory to store results [input file directory]
  -c, --config [CONFIG]
                        YAML configuration file, the default is C:\Users\andrey.vukolov\src\summarizer\config.yml
  -m, --model [MODEL]   Quantized LLM file to load, overrides config file setting
  -cl, --context-length [CONTEXT_LENGTH]
                        LLM context window length limit [4096]
  -b, --batch-length [BATCH_LENGTH]
                        LLM loader batch length [2048]
  -w, --window-coeff [WINDOW_COEFF]
                        Context window width coefficient to calculate the tokenizer context window, window-width = window-coeff * context-length [0.6714], overrides config file setting
  -ov, --overlap-coeff [OVERLAP_COEFF]
                        Context window overlap coefficient [0.05], overrides config file setting
  -pl, --predict-limit-coeff [PREDICT_LIMIT_COEFF]
                        Prediction limit calculation coefficient [0.4272], overrides config file setting
  --model-remap         Set on model RAM remapping mode (for multiple-instance or multi-GPU servers)
  --enable-webui        Turn on web UI renderer on llama.cpp server
  -gl, --gpu-layers [GPU_LAYERS]
                        Instructs the server to upload a number of LLM layers to GPU VRAM [40]
  -p, --prompt-preset [PROMPT_PRESET]
                        Select the prompt preset instead of the default (first appeared) one
  -lp, --list-presets   List prompt presets, then exit
  -j, --join-text       Join all the generated text into one Markdown document
  -wh, --whisper        Use whisper.cpp utility to recognize speech. Requires whisper.cpp and ffmpeg being installed
  -wl, --whisper-language [WHISPER_LANGUAGE]
                        Select the language preset for whisper.cpp
  --whisper-translate   Set translation mode for whisper.cpp
  -wt, --whisper-keep-txt
                        Do not remove TXT artifact from whisper.cpp
  -rm, --render-markdown
                        Render Markdown to HTML (requires Python Markdown extension package installed)
  --video-frames        Tries to extract preview frames from the input video when rendering Markdown to HTML
  --video-frames-cnt [VIDEO_FRAMES_CNT]
                        Number of the preview frames to generate [2..99]
  --video-frames-width [VIDEO_FRAMES_WIDTH]
                        Preview frame width
```

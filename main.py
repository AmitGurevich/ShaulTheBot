import os
import subprocess
from pathlib import Path
from IPython.display import Audio

input_text =  "אַתֶּם הֶאֱזַנְתֶּם לְחַיוֹת כִּיס, הַפּוֹדְקָאסְט הַכַּלְכָּלִי שֶׁל כָּאן."

model_pth_path = Path('tts_model/saspeech_nikud_7350.pth')
model_config_path = model_pth_path.with_name('config_overflow.json')
vocoder_pth_path = Path('hifigan_model/checkpoint_500000.pth')
vocoder_config_path = Path('hifigan_model/config_hifigan.json')

# Where will the outputs be saved?
output_folder = "outputs"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder named {output_folder} created.")
else:
    print(f"Folder named {output_folder} already exists.")

def escape_dquote(s):
    return s.replace('"', r'\"')

global_p = None

def run_model(text, output_wav_path):
    global global_p
    call_tts_string = f"""CUDA_VISIBLE_DEVICES=0 tts --text "{escape_dquote(text)}" \
        --model_path {model_pth_path} \
        --config_path {model_config_path} \
        --vocoder_path {vocoder_pth_path} \
        --vocoder_config_path {vocoder_config_path} \
        --out_path "{output_wav_path}" """
    try:
        print(call_tts_string)
        p = subprocess.Popen(['bash','-c',call_tts_string], start_new_session=True)
        global_p = p
        # throw an exception if the called process exited with an error
        p.communicate(timeout=60)
    except subprocess.TimeoutExpired as e:
        print(f'Timeout for {call_tts_string} (60s) expired', file=sys.stderr)
        print('Terminating the whole process group...', file=sys.stderr)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)

run_model(input_text, output_folder + "/output.wav")

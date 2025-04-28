import json
import os
import sounddevice as sd
import scipy.io.wavfile as wav


class Voice:
    def __init__(self):
        with open('config.json') as f:
            self.config = json.load(f)


    def voice_translate(self):
        def record_audio(duration, output_file):
            """
            录制音频并保存为 MP3 文件
            :param duration: 录音时长（秒）
            :param output_file: 保存的文件名
            """
            try:
                print(f"开始录音... 持续 {duration} 秒")
                fs = 44100  # 采样率
                audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
                sd.wait()  # 等待录音完成
                wav.write(output_file, fs, audio_data)
                print(f"录音完成，保存为 {output_file}")

            except Exception as e:
                print(f"录音失败: {e}")

        def run_whisper(input_file, language, output_dir, output_format, model, task):
            """
            执行 whisper 命令
            :param model_dir: 模型路径
            :param input_file: 音频文件路径
            :param language: 语言代码（如 'zh'）
            :param output_dir: 输出目录
            :param output_format: 输出格式（如 'txt'）
            :param model: 模型类型（如 'medium'）
            :param task: 任务类型（如 'translate'）
            """
            try:
                command = (
                    f"whisper {input_file} --language {language} "
                    f"--output_dir {output_dir} --output_format {output_format} "
                    f"--model {model} --task {task}"
                )
                print(f"运行命令: {command}")
                os.system(command)
                print("Whisper 执行完成")
            except Exception as e:
                print(f"运行 whisper 时出错: {e}")


        # 设置录音时长和文件名
        output_file = self.config["voice_record_path"]

        # 录音并保存为 MP3
        if os.path.exists(output_file):
            os.remove(output_file)
        record_audio(self.config["voice_record_seconds"], output_file)

        # 执行 whisper 命令
        run_whisper(
            input_file=output_file,
            language="zh",
            output_dir=r".",
            output_format="txt",
            model=self.config["voice_model"],
            task="translate"
        )


if __name__ == "__main__":
    voice = Voice()
    voice.voice_translate()
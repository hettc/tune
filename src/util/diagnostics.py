import shutil
import tempfile
import torch.utils.tensorboard as tb
import subprocess

class LogBoard:
    def __init__(self, log_dir: str, port: int = 6006):
        self.log_dir = log_dir
        self.port = port

    def launch(self):
        shutil.rmtree(tempfile.gettempdir() + "/.tensorboard-info", ignore_errors=True) # sort of 'force reload' for tensorboard
        command = [
            'tensorboard',
            '--logdir', self.log_dir,
            '--reload_interval', '1',
            '--port', str(self.port)
        ]
        subprocess.Popen(command)

    def clear(self, folder: str = None):
        if folder is None:
            shutil.rmtree(self.log_dir, ignore_errors=True)
        else:
            shutil.rmtree(f"{self.log_dir}/{folder}", ignore_errors=True)

    def get_logger(self, name: str):
        return tb.SummaryWriter(f"{self.log_dir}/{name}")

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/train_colab.ipynb")


def test_colab_notebook_runs_offline_eval_before_and_after_training():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"]).strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    assert "httpx" in code_cells[0].split()
    upload = code_cells[1]
    assert 'uploaded["eval_prompts.py"]' in upload
    assert 'uploaded["distill.py"]' in upload
    assert 'Path("eval/out").mkdir(parents=True, exist_ok=True)' in upload
    assert code_cells[2] == "!python eval_prompts.py --skip-judge --name base"
    assert code_cells[3] == "!python train_qlora.py"
    assert (
        code_cells[4]
        == "!python eval_prompts.py --skip-judge --adapter out/adapter --name adapter"
    )
    assert 'shutil.make_archive("prompt-lora-run", "zip", staging_dir)' in code_cells[5]
    assert 'files.download("prompt-lora-run.zip")' in code_cells[5]
    assert 'copytree("out/adapter", staging_dir / "out/adapter")' in code_cells[5]
    assert 'glob("*.json")' in code_cells[5]

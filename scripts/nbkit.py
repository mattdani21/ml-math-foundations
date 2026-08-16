"""Shared helper for building Colab notebooks (nbformat).

Usage from a build script in scripts/:
    import nbkit
    cells = [nbkit.md("..."), nbkit.code("..."), ...]
    nbkit.save(cells, "../notebooks/01_Linear_Algebra.ipynb")
"""
import nbformat as nbf


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


def save(cells, path: str) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "colab": {"name": path.rsplit("/", 1)[-1], "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path} ({len(cells)} cells)")

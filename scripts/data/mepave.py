import json
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List
from loguru import logger
import typer


app = typer.Typer()


def dir_iter(folder: Path) -> Iterator[str]:
    valid_extensions = {".jpg"}
    for file in folder.iterdir():
        if file.is_file() and file.suffix in valid_extensions:
            yield file.name
        else:
            logger.warning(f"Not supported file type {file}")


@app.command()
def to_uie(paths: List[str], imgpath: str = typer.Option(...)):
    imgpath: Path = Path(imgpath)
    img_dict = set(dir_iter(imgpath))

    tag_rule = re.compile(r"</?([^>]+)>")

    def labels_mapper(text: str, labels: str) -> List[Dict[str, Any]]:
        open_tags = []
        entities = []
        sent = ""
        last_end = 0
        for match in tag_rule.finditer(labels):
            sent += labels[last_end : match.start()]
            last_end = match.end()
            if match.group(0).startswith("</"):
                tag, start_idx = open_tags.pop()
                assert tag == match.group(1)  # intersection is not allowed
                entities.append({"start": start_idx, "end": len(sent), "type": tag})
            else:
                open_tags.append((match.group(1), len(sent)))
        sent += labels[last_end:]
        assert len(open_tags) == 0
        assert sent == text, f"{sent} vs {text}"
        return entities

    types = set()
    files = [Path(f) for f in paths]
    for file in files:
        if file.suffix != ".txt":
            logger.info(f"Skip {file}")
            continue
        logger.info(f"Processing {file}")
        with file.open("r", encoding="utf8") as rf, file.with_name(
            file.stem + ".uie.jsonl"
        ).open("w", encoding="utf8") as wf:
            for line in rf:
                img, ident, text, label = line.strip().split("\t")
                img = img[2:] + ".jpg"
                assert img in img_dict
                entities = labels_mapper(text, label)
                types.update(entity["type"] for entity in entities)
                wf.write(
                    json.dumps(
                        {
                            "image": img,
                            "id": ident,
                            # "tokens": list(text),     # tokenize in chars
                            "text": text,
                            "entities": entities,
                            "raw_labels": label,
                            "uie": "".join(
                                "({}:{})".format(
                                    ent["type"], text[ent["start"] : ent["end"]]
                                )
                                for ent in entities
                            ),
                            "nl": "\n".join(
                                "{}：{}".format(
                                    ent["type"], text[ent["start"] : ent["end"]]
                                )
                                for ent in entities
                            )
                            if len(entities) != 0
                            else "未找到商品属性",
                        },
                        ensure_ascii=False,
                    )
                )
                wf.write("\n")
    logger.info(f"Get {len(types)} types")


if __name__ == "__main__":
    app()

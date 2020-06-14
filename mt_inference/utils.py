from pathlib import Path
import typer


def respective_file(input: Path, other: Path, extension: str = "png"):
    """
    bla…
    """
    if other is None:
        return

    suffix = f".{extension}"

    if other.is_dir():
        other = other / (input.stem + suffix)
    else:
        if not other.suffix.lower() == suffix:
            typer.secho(
                f"Wrong file suffix for {other}. Expected '{suffix}'",
                fg=typer.colors.YELLOW,
            )

    return other

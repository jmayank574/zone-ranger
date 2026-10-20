"""Entry points for stages that are built in later phases."""


def _todo(stage: str, phase: int) -> None:
    raise SystemExit(f"zone-ranger-{stage} is not implemented yet (planned for Phase {phase}).")


def replay() -> None:
    _todo("replay", 2)


def experiment() -> None:
    _todo("experiment", 5)

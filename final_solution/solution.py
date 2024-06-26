import typing as tp

from .model import Model

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[EntityScoreType]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]

model = None


def init_model():
    global model
    model = Model()


def score_texts(messages: tp.Iterable[str], *args, **kwargs) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages])
    # all messages are shorter than 2048 characters
    """
    if not messages:
        return False
    result = model.forward(messages)
    return result

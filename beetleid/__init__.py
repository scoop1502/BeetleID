"""
BeetleID: identify beetles as SPB (southern pine beetle, Dendroctonus frontalis), Cleridae
(checkered beetles) or other_Coleoptera.

    from beetleid import BeetleClassifier          # beetle model only: give it specimen crops
    from beetleid.foundation import pipeline       # BeetleID's own detect -> align -> order pipeline

BeetleClassifier is what other projects use as a plug-in (e.g. ArthropodModel's MCP server calls it
for specimens it classifies as Coleoptera).  It expects pose-aligned specimen crops like those the
pipeline makes (RGB numpy arrays).
"""
from .classifier import BeetleClassifier, BEETLE_MODEL, BEETLE_NAMES   # noqa: F401

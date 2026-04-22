from __future__ import annotations

from .base import BaseProcessorMixin
from .download import DownloadMixin
from .processing_steps import ProcessingStepsMixin
from .workflow import TQDM_AVAILABLE as WORKFLOW_TQDM_AVAILABLE, WorkflowMixin

TQDM_AVAILABLE = WORKFLOW_TQDM_AVAILABLE


class TelegramProcessor(
    WorkflowMixin,
    ProcessingStepsMixin,
    DownloadMixin,
    BaseProcessorMixin,
):
    """Main processor composed from smaller behavior-focused mixins."""

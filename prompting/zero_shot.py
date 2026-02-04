from .base import PromptFramework, Message

class ZeroShotPrompting(PromptFramework):

    @property
    def name(self) -> str:
        return "zero_shot"

    def _build_messages(self, task, **kwargs):
        vars = self._base_vars(task, **kwargs)
        user_prompt = self.renderer.render("zero_shot.txt", **vars)

        return [
            self._system_message(),
            Message(role="user", content=user_prompt),
        ]

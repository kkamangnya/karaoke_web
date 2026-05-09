# karaoke_web Work Log

Format: `YYYY-MM-DD HH:mm:ss +09:00 | actor | action | details`

## Timeline
- `2026-05-10 05:50:04 +09:00` | Codex | Work log initialized | Added a repository-level timeline file for future task notes.
- `2026-05-10 05:50:04 +09:00` | Codex | UI rollback | Reverted the popover positioning change in the browser karaoke pitch shifter after an error was reported.
- `2026-05-10 05:51:00 +09:00` | Codex | Publish to GitHub | Copied the browser and Python karaoke pitch shifter files into the `karaoke_web` repository and prepared them for commit.
- `2026-05-10 05:56:00 +09:00` | Codex | Popover fix | Updated help popovers to use fixed positioning with viewport clamping so the tooltip text is not hidden by nearby panels or window edges.
- `2026-05-10 05:58:00 +09:00` | Codex | Popover portal | Moved help text popovers to a single body-level overlay so the processing panel cannot obscure them.

## Notes
- Future changes should append new entries at the top of the timeline.
- Keep entries short, timestamped, and concrete so the repo shows a clear activity history.

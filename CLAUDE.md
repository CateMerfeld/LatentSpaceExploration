# Session naming and checkpointing

- Early in a chat, once it's clear the conversation involves nontrivial or multi-step work, ask the user: "Is this chat worth naming so you can find it again later?"
- If they say yes and give a name `<name>`:
  - Tell them to run `/rename <name>` themselves — there is no tool available to set the Claude Code session name programmatically, so this step can't be automated away.
  - Create `.claude/session-logs/<name>.tex` and write a short summary of the conversation so far (goal, key decisions, current state).
- At natural checkpoints thereafter (finishing a task, wrapping up a nontrivial chunk of work — not on a fixed timer), update `.claude/session-logs/<name>.tex` with a fresh concise summary: what's done, key decisions/tradeoffs, open questions, and next steps. Overwrite rather than append — it should stay a short "resume-from-here" snapshot, not a transcript.
- When resuming a named session, read `.claude/session-logs/<name>.tex` first instead of re-deriving context from scratch.

When starting a new experiment ask the user if a report should be written to detail what was done. Ask before beginning the experiment or performing large tasks for the experiment. If they say yes create a report in the experiments directory. The report should be a `.tex` file. The report should start with an overview that is readable and can be informal. The rest of the report should be written formally, but should be as readable as possible. Avoid inserting file or figure names into sentences.

Whenever possible submit slurm jobs rather than running on cpu. Run the job on GPU rather than CPU.

Put files in places that make sense logically. You can make new directories if needed.

Unless explicitly instructed, never put hashes on PCA plot axes.  

Plot titles, labels, legends, etc need to be big enough to see when the plot is included in a paper. Something like title (18pt), axis labels (16pt), tick labels (13pt), and legend text (13pt) is a good starting place but feel free to go bigger if it makes sense.
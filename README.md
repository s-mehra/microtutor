# about
I love to teach. I even have dreams of opening up a brick and mortar school when I'm older. However I've been thinking increasingly of what education is going to look like in an AI driven world. Models and applications can already give you all the information in the world. But there are few systems that can understand your goals, why you want to learn, constantly track your progress, and adapt to your understanding. This is my attempt at building an AI tutor, who helps you learn AND understand anything you want at a deep level from the ground up


# microtutor

A knowledge-state aware AI tutor that remembers what you know, tracks what you struggle with, and sequences material accordingly.

Most AI tutors are stateless - every conversation starts from zero. Microtutor is different. It remembers what you've mastered, what you're struggling with, and what you haven't seen yet. It builds a personalized curriculum around your goals, teaches you with real explanations and worked examples, and adapts in real time based on how you're doing. Come back a week later and it picks up exactly where you left off.

## Getting started

Requires Python 3.11+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/s-mehra/microtutor.git
cd microtutor
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
microtutor
```

## What happens when you launch

1. **First run** - the app asks for your name and API key, which are saved locally to `~/.microtutor/config.json`. You only do this once.

2. **New course** - a diagnostic conversation figures out what you want to learn, what you already know, and how deep you want to go. The tutor probes your existing knowledge with specific questions, not just "what's your background." 

3. **Lessons** - each lesson follows a structured flow: prerequisite review, teaching with formal definitions and examples, interactive Q&A, and assessment. The tutor adapts based on your responses. Mastery scores update after every interaction.

4. **Between sessions** - your progress is saved automatically. Mastery decays over time without practice, so the tutor will circle back to concepts you haven't touched in a while. Lesson notes are saved as markdown files you can review offline.

5. **Multiple courses** - you can have several courses running at once. On launch, pick which one to continue or start a new one.

## License

MIT

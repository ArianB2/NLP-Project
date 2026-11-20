# Labeling guidelines (v1)

**Question for every row:** Is this message hostile toward another player (or a group of players)?

- **1 = hostile toward a player.** The speaker insults, mocks, blames, threatens, or tries to drive off another player or the team.
- **0 = not hostile toward a player.** Everything else.

Read the context lines first. `self` = the same player, `ally` = a teammate, `enemy` = the other team. Then label the **message** only. The context helps you work out who the message is aimed at, but a hostile context line does not make a calm reply hostile.

## Label 1 (hostile)

| Type | Example |
|---|---|
| Direct insult, with or without swearing | "mid noob", "u are trash", "fuck off sf" |
| Blame aimed at a player | "u lost us the game necro" |
| Mocking or sarcasm aimed at a player | "great ult genius", "whine more guys" |
| Telling a player to leave or stop talking | "uninstall", "shut up" |
| Threats, including in-game ones aimed at the person | "i will report u and ruin ur game" |
| Report request **with** an insult | "report mid plz noooob" |
| "ez" or "gg ez" said **to** the other team as a taunt | "ez noobs" |

## Label 0 (not hostile)

| Type | Example |
|---|---|
| Swearing at the game, Valve, lag, or a hero/creep/tree | "stupid valve", "my shit is lagging", "fucking tree" |
| Swearing at yourself | "fuck i missed", "im so bad" |
| Plain surprise or frustration | "wtf", "damn", "LOL WTF" |
| Friendly banter between teammates when the context shows it's friendly | "noob :D" after "lol nice try" |
| Game talk and requests without insults | "report mid" (no insult), "pause pls", "go rosh" |
| Neutral or positive chat | "gg wp", "ty", "nice" |

## Tie-breakers

1. **Who is the target?** If it's a player or the team, lean 1. If it's the game, Valve, the speaker, or nobody, it's 0.
2. **Profanity alone is never enough** for a 1. "fuck" by itself is 0.
3. **No profanity is not a reason for 0.** "nice brain" said to a feeding teammate is 1.
4. **Can't tell?** Put an `x` in the unsure column and choose your best guess. If you truly can't guess, leave the label blank and it will be skipped.
5. **Non-English messages:** label them only if you understand them. Otherwise leave blank and mark unsure.

## Process

1. Both of us label the same 50 shared rows first (they are mixed into each sheet), without talking about them.
2. Run `python src/agreement.py` to get Cohen's kappa and the list of disagreements.
3. Talk through each disagreement, fill in `final_label`, and update these guidelines if a new rule came up (bump to v2).
4. Then label the rest of your own sheet.

Do not open `data/labels/sample_key.csv` while labeling. It shows the CONDA label for each row.

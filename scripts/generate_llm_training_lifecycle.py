"""Render the conceptual training lifecycle: python scripts/generate_llm_training_lifecycle.py.

Requires matplotlib; shares the handbook's diagram primitives and export checks.
"""
import sys

sys.dont_write_bytecode = True

from generate_attention_diagrams import canvas, text, box, arrow, panel, save, C


def lifecycle():
    fig, ax = canvas(
        'LLM training: regimes within an iterative development lifecycle',
        '[To be reviewed]  Conceptual map; reports name boundaries differently. Paths and loops are optional.',
        height=1450,
    )
    # The bypass makes clear that a distinct mid-training phase is not required.
    arrow(ax, [(300, 235), (300, 175), (1500, 175), (1500, 235)],
          color='line', dashed=True)
    text(ax, 900, 148, 'A distinct mid-training run may be skipped',
         size=14, color=C['muted'], ha='center')
    panel(ax, 50, 235, 500, 465, '1  BROAD PRE-TRAINING',
          'Broad language, knowledge, code and reasoning')
    panel(ax, 650, 235, 500, 465, '2  CONTINUED TRAINING',
          'Optional mid-training / late pre-training')
    panel(ax, 1250, 235, 500, 465, '3  POST-TRAINING',
          'Shape behavior / responses / actions')
    box(ax, 85, 360, 430, 120,
        'Large mixed corpora\nUsually next-token prediction\nFIM / auxiliary targets may coexist',
        size=15)
    box(ax, 85, 535, 430, 90, 'Broad base checkpoint\nQuality and mixture matter from the start',
        fill='mint', edge='green', size=14)
    arrow(ax, [(300, 480), (300, 535)], color='blue')
    box(ax, 685, 360, 430, 120,
        'Domain / capability emphasis\nReasoning-rich data\nLong-context adaptation', size=15)
    box(ax, 685, 535, 430, 90, 'Often continued LM loss\nDefined by purpose / data transition',
        fill='mint', edge='green', size=14)
    arrow(ax, [(900, 480), (900, 535)], color='blue')
    arrow(ax, [(550, 435), (650, 435)], color='blue')
    arrow(ax, [(1150, 435), (1250, 435)], color='blue')
    for y, label, fill, edge in [
        (345, 'SFT: instruction / CoT / tool-agent targets', 'pale', 'blue'),
        (420, 'Direct preferences: e.g. offline DPO', 'pale', 'blue'),
        (495, 'Rollout RL: feedback + chosen update rule', 'pale', 'blue'),
        (570, 'Teacher feedback: offline / OPD / OPSD', 'cache', 'purple'),
    ]:
        box(ax, 1280, y, 440, 56, label, fill=fill, edge=edge, size=13)
    text(ax, 1500, 663, 'Compatible routes; not four required steps',
         size=13, ha='center', color=C['muted'])

    panel(ax, 50, 755, 1700, 270, 'EXAMPLE POST-TRAINING LOOP',
          'SFT can follow RL again; direct preferences and teacher feedback can enter at different points.')
    box(ax, 100, 865, 350, 66, 'SFT / refreshed checkpoint', size=17)
    box(ax, 650, 865, 280, 66, 'RL', fill='mint', edge='green', size=19)
    box(ax, 1110, 865, 560, 66, 'Generate / collect + verify new training data',
        fill='sand', edge='amber', size=16)
    arrow(ax, [(450, 898), (650, 898)], color='blue')
    arrow(ax, [(930, 898), (1110, 898)], color='blue')
    arrow(ax, [(1390, 931), (1390, 985), (275, 985), (275, 931)],
          color='amber', dashed=True)
    text(ax, 830, 963, 'New verified demonstrations / targets feed a later SFT round',
         size=13, ha='center', color=C['amber'])
    # This connection associates the example loop with behavioral post-training.
    arrow(ax, [(1500, 700), (1500, 755)], color='blue')

    box(ax, 50, 1070, 1700, 95,
        'CROSS-CUTTING TRACKS THROUGH DEVELOPMENT\n'
        'Data quality / decontamination  ·  Safety behavior  ·  Independent evaluation / retention  ·  Compute / efficiency',
        fill='#eef1f5', edge='line', size=16)
    box(ax, 50, 1210, 1700, 155, fill='cache', edge='purple')
    text(ax, 900, 1245, 'RECURRING WORKSTREAM: specialized adaptation + release / serving',
         size=20, ha='center', color=C['purple'], weight='bold')
    text(ax, 900, 1292,
         'Domain/tool adaptation, compact students and behavioral hardening reuse the training routes above.',
         size=15, ha='center')
    text(ax, 900, 1333,
         'Runtime kernels, cache layouts and serving are execution choices; observed failures can inform later training.',
         size=14, ha='center')
    text(ax, 60, 1410,
         'This workstream is not a mandatory fourth training objective. Deployment does not automatically update model weights.',
         size=14, color=C['muted'])
    save(fig, ax, 'llm-training-lifecycle',
         'Conceptual LLM lifecycle with three training regimes, an optional mid-training bypass, '
         'an illustrative SFT-RL-verified-data loop, cross-cutting quality/safety/evaluation/efficiency, '
         'and recurring specialization and release/serving work.')


if __name__ == '__main__':
    lifecycle()
    print('Generated llm-training-lifecycle.png and .svg in figs/.')

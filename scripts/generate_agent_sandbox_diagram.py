"""Render the sandbox trust-boundary schematic; requires matplotlib.

Run: python scripts/generate_agent_sandbox_diagram.py
"""
import sys

sys.dont_write_bytecode = True

from generate_attention_diagrams import canvas, text, box, arrow, panel, save, C


def sandbox_boundary():
    fig, ax = canvas(
        'Agent sandbox: isolate execution, authorize actions, grade independently',
        '[To be reviewed]  Conceptual coding-agent design; policy and runtime choices must be enforced outside candidate code.',
        height=1380,
    )
    text(ax, 50, 190, 'MODEL / AGENT', 20, weight='bold')
    box(ax, 50, 250, 420, 110,
        'Propose a tool call\nArguments are untrusted input', size=17)
    text(ax, 550, 190, 'TRUSTED CONTROL PLANE', 20,
         color=C['green'], weight='bold')
    box(ax, 550, 250, 500, 110,
        'Policy / tool broker\nValidate identity, scope, arguments, budget',
        fill='mint', edge='green', size=16)
    box(ax, 550, 425, 500, 86,
        'Externally defined policy / scoped interfaces\nNo inherited general-purpose host authority',
        fill='#ffffff', edge='green', size=14)
    arrow(ax, [(800, 425), (800, 360)], color='green')
    arrow(ax, [(470, 305), (550, 305)], color='blue')
    arrow(ax, [(1050, 305), (1240, 305)], color='green')

    # The actor-owned worker and the evaluation worker are distinct zones.
    panel(ax, 1200, 175, 550, 535, 'CONSTRAINED WORKER',
          'Generated code / repository scripts remain untrusted')
    box(ax, 1240, 290, 470, 100,
        'Execute code, builds and public tests\nInside the selected isolation boundary',
        fill='sand', edge='amber', size=15)
    box(ax, 1240, 430, 470, 85,
        'Scoped input + writable workspace\nNo unrelated host mounts or secrets',
        fill='sand', edge='amber', size=15)
    box(ax, 1240, 555, 470, 85,
        'Controlled network / CPU / memory / tasks\nExternal wall deadline + output / disk limits',
        fill='sand', edge='amber', size=14)
    box(ax, 550, 565, 500, 110,
        'Bounded observations + typed status\nTrusted metadata; workload text stays data',
        fill='pale', edge='blue', size=16)
    arrow(ax, [(1200, 650), (1110, 650), (1110, 620), (1050, 620)], color='blue')
    arrow(ax, [(550, 620), (260, 620), (260, 360)], color='blue')
    text(ax, 400, 590, 'Return as observations', 13,
         color=C['blue'], ha='center')

    box(ax, 1200, 765, 550, 80,
        'Capture the declared artifact + provenance\nDo not accept a self-reported score',
        fill='mint', edge='green', size=15)
    arrow(ax, [(1475, 710), (1475, 765)], color='green')
    panel(ax, 50, 900, 1700, 330, 'SEPARATE EVALUATION INFRASTRUCTURE',
          'The actor cannot rewrite grading policy or authoritative results; submitted code is untrusted during evaluation too.')
    box(ax, 95, 1010, 435, 115,
        'Trusted evaluation controller\nFixed grading protocol\nAuthoritative result store',
        fill='mint', edge='green', size=16)
    box(ax, 795, 1010, 900, 115,
        'Fresh constrained evaluation worker\nBase snapshot + submitted artifact + declared test protocol\nExecute candidate code here, not inside the privileged controller',
        fill='sand', edge='amber', size=16)
    arrow(ax, [(1475, 845), (1475, 1010)], color='green')
    arrow(ax, [(530, 1068), (795, 1068)], color='green')
    text(ax, 662, 1034, 'Start evaluation', 13, ha='center', color=C['green'])
    arrow(ax, [(1245, 1125), (1245, 1180), (312, 1180), (312, 1125)], color='blue')
    text(ax, 810, 1158, 'Collect results through the defined grading channel',
         size=13, ha='center', color=C['blue'])
    text(ax, 60, 1285,
         'Isolation does not replace semantic authorization. A disposable workspace does not undo remote side effects.',
         size=15, weight='bold')
    text(ax, 60, 1338,
         'This is a boundary schematic, not a runnable security configuration or proof that an evaluator is correct.',
         size=14, color=C['muted'])
    save(fig, ax, 'agent-sandbox-boundary',
         'An untrusted agent request is validated by a trusted broker, runs in a constrained worker, '
         'returns bounded observations, and submits an artifact to separate constrained evaluation '
         'under an independently controlled grading protocol.')


if __name__ == '__main__':
    sandbox_boundary()
    print('Generated agent-sandbox-boundary.png and .svg in figs/.')

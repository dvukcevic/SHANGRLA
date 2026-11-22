# Starter script for interactively exploring the IRV AWAIRE ballot-level
# comparison workflow example.

from examples.workflow_irv_awaire_ballot_comparison import create_workflow_data

# Run the workflow setup to inspect the contest, reported CVRs, and paired sample.
audit, contests, cvrs, mvr_sample, cvr_sample = create_workflow_data()

# The AWAIRE assertions are leaves in a composition tree of alternative orders.
assertions = contests["mayor"].assertions
root = assertions.registry.root

# Inspect the setup, for example:
#   contests["mayor"].assertion_strategy
#   list(assertions.assertions)[:5]
#   cvrs[:3], mvr_sample[:3], cvr_sample[:3]
#   root.children[:2]

# To calculate evidence and print the composed audit status:
#   assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)
#   audit.summarize_status(contests)

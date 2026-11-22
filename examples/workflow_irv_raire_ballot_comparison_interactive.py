# Starter script for interactively exploring the IRV RAIRE ballot comparison
# workflow example.

from examples.workflow_irv_raire_ballot_comparison import create_workflow_data

# Run the workflow setup to get the data structures for inspection.
audit, contests, cvrs, mvr_sample, cvr_sample = create_workflow_data()

# Extract the assertions for the contest, for easy access.
assertions = contests["mayor"].assertions

# Inspect the setup, for example:
#   contests["mayor"].assertion_json
#   assertions.assertions["Alice v Bob"].margin
#   cvrs[:3], mvr_sample[:3], cvr_sample[:3]

# To calculate evidence and print the audit status:
#   assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)
#   audit.summarize_status(contests)

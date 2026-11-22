# This is a starter script for interactively exploring the plurality ballot
# comparison workflow example.

from examples.workflow_plurality_ballot_comparison import create_workflow_data

# Run the workflow setup to get the data structures for inspection.
audit, contests, cvrs, mvr_sample, cvr_sample = create_workflow_data()

# Extract the assertions for the contest, for easy access.
assertions = contests["mayor"].assertions

# Inspect the setup, for example:
#   contests["mayor"].tally
#   assertions.assertions["Alice v Bob"].margin
#   mvr_sample[0], cvr_sample[0]

# To calculate evidence and print the audit status:
#   assertions.set_p_values(mvr_sample, cvr_sample, use_all=True)
#   audit.summarize_status(contests)

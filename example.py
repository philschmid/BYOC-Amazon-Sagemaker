from sagemaker.huggingface import HuggingFaceModel
import sagemaker

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='<YOUR_ROLE_NAME>')['Role']['Arn']

env = {
	'PREDICT_ROUTE':'/invocations',
	'HEALTH_ROUTE':'/ping'
}

# create Model Class
infinity_model = HuggingFaceModel(
	image_uri="379486364332.dkr.ecr.us-east-1.amazonaws.com/infinity-trial:{TAG}",
	env=env,
	role=role, 
)

# deploy model to SageMaker Inference
predictor = infinity_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.g4dn.xlarge' # ec2 instance type
)

# run request
predictor.predict({
	'inputs': "Can you please let us know more details about your "
})

#{'vector': [0.3552246,
#  -0.21740723,
#  0.5498047,
#  0.52001953,
#  1.2236328,
#  -0.51660156,
#  -0.09820557,
#  0.2680664,
#  0.14953613,
#  0.11859131,
#  0.035064697,
#  -0.5361328,

from aws_cdk import App, Stack, Duration
from constructs import Construct
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam

class S3CleanupStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        bucket_name = "kosokolovsky-projects"

        lambda_role = iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:ListBucket",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            resources=[
                f"arn:aws:s3:::{bucket_name}",
                f"arn:aws:s3:::{bucket_name}/*"
            ]
        ))

        cleanup_lambda = lambda_.Function(
            self, "S3CleanupLambda",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_asset('lambda_function'),
            environment={
                "S3_BUCKET_NAME_PROJECTS": bucket_name,
            },
            role=lambda_role 
        )

        rule = events.Rule(
            self, "Rule",
            schedule=events.Schedule.rate(Duration.days(30)),
        )
        rule.add_target(targets.LambdaFunction(cleanup_lambda))

app = App()
S3CleanupStack(app, "S3CleanupStack", env={'region': 'eu-central-1'})
app.synth()


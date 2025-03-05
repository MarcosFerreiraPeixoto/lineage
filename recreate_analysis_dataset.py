def create_analysis_copy(source_analysis_id, new_analysis_id, new_analysis_name):
    # Get original analysis definition
    analysis_def = get_analysis_definition(source_analysis_id)
    if not analysis_def:
        raise ValueError(f"Analysis {source_analysis_id} not found")

    # Initialize QuickSight client
    quicksight = boto3.client('quicksight', region_name=AWS_REGION)
    aws_account_id = boto3.client('sts').get_caller_identity()['Account']

    try:
        response = quicksight.create_analysis(
            AwsAccountId=aws_account_id,
            AnalysisId=new_analysis_id,
            Name=new_analysis_name,
            SourceEntity=analysis_def['SourceEntity'],  # Critical structure
            ThemeArn=analysis_def.get('ThemeArn', ''),
            Tags=analysis_def.get('Tags', []),
            Permissions=[
                {
                    'Principal': f'arn:aws:iam::{aws_account_id}:user/your-user',
                    'Actions': ['quicksight:UpdateAnalysisPermissions']
                }
            ]
        )
        return response
    except quicksight.exceptions.ResourceExistsException:
        print(f"Analysis {new_analysis_id} already exists")
        return None
    

# Example usage
create_analysis_copy(
    source_analysis_id='original-analysis-id',
    new_analysis_id='new-analysis-id',
    new_analysis_name='Copied Analysis'
)

def recreate_dataset(source_dataset_id, new_dataset_id, new_name, permissions=None):
    """
    Recreates a QuickSight dataset with optional ARN remapping
    
    :param arn_remap: Dictionary of old ARN -> new ARN replacements
    :param permissions: List of permission dictionaries for the new dataset
    """
    quicksight = boto3.client('quicksight')
    aws_account_id = boto3.client('sts').get_caller_identity()['Account']
    
    # Get original dataset configuration
    dataset_def = get_dataset_definition(source_dataset_id)
    if not dataset_def:
        raise ValueError(f"Dataset {source_dataset_id} not found")

    # Prepare base configuration
    config = {
        'AwsAccountId': aws_account_id,
        'DataSetId': new_dataset_id,
        'Name': new_name,
        'PhysicalTableMap': dataset_def.get('PhysicalTableMap', {}),
        'LogicalTableMap': dataset_def.get('LogicalTableMap', {}),
        'ImportMode': dataset_def.get('ImportMode', 'SPICE'),
        'ColumnGroups': dataset_def.get('ColumnGroups', []),
        'FieldFolders': dataset_def.get('FieldFolders', {}),
        'RowLevelPermissionDataSet': dataset_def.get('RowLevelPermissionDataSet', {}),
        'DataSetUsageConfiguration': dataset_def.get('DataSetUsageConfiguration', {})
    }


    # Set permissions (default to original if not specified)
    if not permissions and 'Permissions' in dataset_def:
        config['Permissions'] = [
            {
                'Principal': p['Principal'],
                'Actions': p['Actions']
            } for p in dataset_def['Permissions']
        ]
    else:
        config['Permissions'] = permissions or []

    try:
        response = quicksight.create_data_set(**config)
        print(f"Created dataset {new_dataset_id} successfully")
        return response
    except quicksight.exceptions.ResourceExistsException:
        print(f"Dataset {new_dataset_id} already exists")
        return None